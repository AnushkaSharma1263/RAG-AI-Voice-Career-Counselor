"""Voice career counselor on Streamlit.
mic (st.audio_input) -> Groq Whisper -> FAISS retrieval -> streamed Groq LLM -> Orpheus TTS per chunk (parallel)
-> chunks pushed to a browser-side queue as soon as each is ready, so speech starts before the reply is finished."""
import base64
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
import streamlit.components.v1 as components
from groq import Groq

import counselor as C
from rag import KnowledgeBase, format_context

st.set_page_config(page_title="Career counselor", page_icon="🎙️", layout="centered")

st.markdown(
    """<style>
    h1 {letter-spacing:-0.03em; font-weight:800}
    /* collapse the invisible audio-player elements so they take no space */
    div[data-testid="stElementContainer"]:has([data-mhq]), div[data-testid="stElementContainer"]:has(iframe[height="0"])
      {position:absolute; height:0; margin:0; overflow:hidden}
    </style>""",
    unsafe_allow_html=True,
)


# ---------- resources (created once per server process) ----------
def _api_key() -> str | None:
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.getenv("GROQ_API_KEY")


@st.cache_resource
def get_client(key: str) -> Groq:
    return Groq(api_key=key)


@st.cache_resource(show_spinner="Loading knowledge base…")
def get_kb() -> KnowledgeBase:
    kb = KnowledgeBase()
    kb.build()
    return kb


@st.cache_resource
def get_pool() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=4)  # parallel TTS; stays under Orpheus rate limits


key = _api_key()
if not key:
    st.error("Add GROQ_API_KEY to the app's secrets (Settings → Secrets) and reboot.")
    st.stop()
client, kb, pool = get_client(key), get_kb(), get_pool()

# ---------- browser-side playback queue ----------
# Each clip runs a tiny script that pushes audio into a queue on the page, which plays clips back-to-back.
# Because the queue lives on the page, playback continues after Streamlit re-renders.
PLAYER_JS = """<span data-mhq></span><script>
(function () {
  const P = window.frameElement ? window.parent : window;  // st.html runs in the page; iframe fallback uses parent
  const Q = (P.__mhq = P.__mhq || { q: [], seen: {}, playing: false });
  if (!Q.next) {
    Q.next = function () {
      if (Q.playing) return;
      const url = Q.q.shift();
      if (!url) return;
      Q.playing = true;
      const a = new P.Audio(url); Q.cur = a;
      const done = () => { Q.playing = false; Q.next(); };
      a.onended = done; a.onerror = done; a.play().catch(done);
    };
  }
  const id = "__ID__"; if (Q.seen[id]) return; Q.seen[id] = 1;
  __ACTION__
})();
</script>"""


def run_js(html: str):
    try:
        st.html(html, unsafe_allow_javascript=True)  # current API: runs in the page, no iframe
    except TypeError:
        components.html(html, height=0)  # older Streamlit: iframe with same-origin access


def play_clip(wav: bytes, clip_id: str):
    b64 = base64.b64encode(wav).decode()
    action = f'Q.q.push("data:audio/wav;base64,{b64}"); Q.next();'
    run_js(PLAYER_JS.replace("__ID__", clip_id).replace("__ACTION__", action))


def reset_player(turn_id: int):
    action = "Q.q.length = 0; Q.playing = false; if (Q.cur) { Q.cur.onended = Q.cur.onerror = null; Q.cur.pause(); }"
    run_js(PLAYER_JS.replace("__ID__", f"reset-{turn_id}").replace("__ACTION__", action))


# ---------- state ----------
ss = st.session_state
ss.setdefault("history", [])  # [{"role", "content"}]
ss.setdefault("turn", 0)
ss.setdefault("last_audio", None)
ss.setdefault("replay", None)
ss.setdefault("timing", None)
ss.setdefault("sources", None)

# ---------- UI ----------
st.title("Talk through your career.")
st.write(
    "Tell me what you study, what you enjoy and what you want from work. "
    "I'll suggest paths that fit and ask what I still need to know."
)
audio = st.audio_input("Tap the mic, speak, then tap stop to send", key="mic")
typed = st.chat_input("…or type your question")

for m in ss.history:
    with st.chat_message("user" if m["role"] == "user" else "assistant"):
        st.write(m["content"])

# ---------- one conversational turn ----------
wav, text = None, typed
if audio is not None:
    data = audio.getvalue()
    h = hashlib.md5(data).hexdigest()
    if h != ss.last_audio:  # reruns re-deliver the same recording; only process it once
        ss.last_audio, wav = h, data

if wav or text:
    ss.turn += 1
    tid = ss.turn
    reset_player(tid)  # a new turn interrupts any audio still playing
    t0 = time.perf_counter()
    ms = lambda: int((time.perf_counter() - t0) * 1000)  # noqa: E731
    tm: dict = {}

    if wav:
        with st.spinner("Listening…"):
            text = C.transcribe(client, wav)
        tm["speech-to-text"] = ms()

    if not text:
        st.warning("I didn't catch that. Try again a bit closer to the mic.")
    else:
        with st.chat_message("user"):
            st.write(text)

        t_r = ms()
        hits = kb.search(C.retrieval_query(ss.history, text))
        tm["retrieval"] = ms() - t_r
        messages = C.build_messages(ss.history, format_context(hits), text)

        reply, buf, early = "", "", True
        futures: list = []
        pos = {"n": 0}  # next clip to render

        with st.chat_message("assistant"):
            slot = st.empty()
            clips = st.container()

            def flush(wait: bool = False):
                """Render finished TTS clips in order; each one starts playing in the browser immediately."""
                while pos["n"] < len(futures) and (wait or futures[pos["n"]].done()):
                    n = pos["n"]
                    try:
                        clip = futures[n].result()
                    except Exception:
                        clip = None
                    if clip:
                        with clips:
                            play_clip(clip, f"{tid}-{n}")
                        tm.setdefault("first audio", ms())
                    futures[n] = clip
                    pos["n"] += 1

            def speak(chunk: str):
                if chunk.strip():
                    futures.append(pool.submit(C.synth, client, chunk.strip()))  # TTS runs while LLM keeps streaming

            stream = client.chat.completions.create(
                model=C.LLM_MODEL, messages=messages, stream=True, max_tokens=260, temperature=0.6
            )
            for ev in stream:
                delta = ev.choices[0].delta.content if ev.choices else None
                if not delta:
                    continue
                tm.setdefault("first token", ms())
                reply += delta
                buf += delta
                slot.markdown(reply + " ▌")
                while (cut := C.next_cut(buf, early)) is not None:
                    speak(buf[:cut])
                    buf, early = buf[cut:], False
                flush()
            slot.markdown(reply)
            speak(buf)
            flush(wait=True)

        tm["total"] = ms()
        ss.history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
        ss.timing = tm
        ss.sources = [h["title"] for h in hits]
        ok = [c for c in futures if isinstance(c, (bytes, bytearray))]
        ss.replay = C.concat_wavs(ok) if ok else None

# ---------- footer: replay, sources, timings ----------
if ss.replay:
    with st.expander("Replay last answer"):
        st.audio(ss.replay, format="audio/wav")
if ss.sources:
    st.caption("Grounded in: " + " · ".join(ss.sources))
if ss.timing:
    st.caption("Latency (ms): " + " · ".join(f"{k} {v}" for k, v in ss.timing.items()))
if ss.history and st.button("Start over"):
    for k in ("history", "replay", "timing", "sources"):
        ss.pop(k, None)
    st.rerun()
