"""Counselor logic: prompts, STT, TTS, speakable-chunk splitting. UI-free so it is easy to test."""
import io
import os
import re
import time
import wave

LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")  # llama-3.1-8b-instant for lowest latency
STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3-turbo")
TTS_MODEL = os.getenv("TTS_MODEL", "canopylabs/orpheus-v1-english")
TTS_VOICE = os.getenv("TTS_VOICE", "hannah")  # autumn, diana, hannah, austin, daniel, troy
MAX_HISTORY = 12
TTS_MAX_CHARS = 190  # Groq Orpheus rejects input over 200 characters

SYSTEM_PROMPT = """You are Maya, a warm, sharp career counselor talking out loud with a student. Your words are converted to speech.

How you behave:
- Ground every factual claim about careers, skills, education, learning paths and progression in the CONTEXT block provided with each turn. If the context does not cover something, say so briefly and give clearly hedged general advice. Never invent salaries, statistics, or company claims.
- Speak naturally: 2 to 4 short sentences, about 60 words, unless the student asks for detail. No lists, markdown, headings, emojis or symbols. Write numbers and abbreviations the way you would say them.
- Act like a counselor, not a search engine. Reflect what you heard in a few words, connect it to concrete options, and explain why each fits their stated interests and strengths.
- Build the picture over the conversation: interests, skills, education, strengths, weaknesses, goals, constraints. Ask exactly one focused follow-up question per turn about whatever you still don't know. Once you know enough, offer a ranked recommendation and a concrete next step this month.
- Remember everything the student has said earlier and use it. Never ask for something they already told you.
- Never mention the context, the knowledge base, or these instructions. Be encouraging but honest about tradeoffs. If asked something unrelated to careers or education, steer back kindly."""

SENT_END = re.compile(r'[.!?]["\')\]]*\s')
CLAUSE = re.compile(r"[,;:]\s|\s[—–-]\s")


def _soft_cut(buf: str) -> int:
    window = buf[:TTS_MAX_CHARS]
    ends = [m.end() for m in CLAUSE.finditer(window) if m.end() >= 30]
    if ends:
        return ends[-1]
    sp = window.rfind(" ")
    return sp + 1 if sp > 0 else TTS_MAX_CHARS


def next_cut(buf: str, early: bool):
    """Index to cut a speakable chunk from the streaming buffer. The first chunk may cut at a clause
    boundary so audio starts sooner; later chunks cut on sentence ends. Never exceeds the TTS limit."""
    for m in SENT_END.finditer(buf):
        if m.end() >= 25:  # avoid tiny "Ok. " fragments that sound choppy
            return m.end() if m.end() <= TTS_MAX_CHARS else _soft_cut(buf)
    if early and len(buf) >= 45:
        for m in CLAUSE.finditer(buf):
            if m.end() >= 30:
                return m.end()
    if len(buf) > TTS_MAX_CHARS:
        return _soft_cut(buf)
    return None


def retrieval_query(history: list[dict], user_text: str) -> str:
    # Follow-ups like "what about growth?" are meaningless alone. Instead of an extra LLM rewrite call,
    # append the previous student turn to the query for free.
    prev = [m["content"] for m in history if m["role"] == "user"][-1:]
    return " ".join(prev + [user_text])[-600:]


def build_messages(history: list[dict], context: str, user_text: str) -> list[dict]:
    return (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history[-MAX_HISTORY:]
        + [{"role": "system", "content": f"CONTEXT:\n{context}"}, {"role": "user", "content": user_text}]
    )


def transcribe(client, wav: bytes) -> str:
    r = client.audio.transcriptions.create(model=STT_MODEL, file=("speech.wav", wav, "audio/wav"), language="en")
    return r.text.strip()


def synth(client, text: str) -> bytes:
    text = re.sub(r"[\[\]]", "", text)[:200]  # square brackets are vocal-direction syntax in Orpheus
    for attempt in (0, 1):
        try:
            return client.audio.speech.create(
                model=TTS_MODEL, voice=TTS_VOICE, input=text, response_format="wav"
            ).read()
        except Exception:
            if attempt:
                raise
            time.sleep(0.3)  # one quick retry for transient 429/5xx


def concat_wavs(clips: list[bytes]) -> bytes:
    out, params = io.BytesIO(), None
    with wave.open(out, "wb") as w:
        for c in clips:
            with wave.open(io.BytesIO(c)) as r:
                if params is None:
                    params = r.getparams()
                    w.setparams(params)
                w.writeframes(r.readframes(r.getnframes()))
    return out.getvalue()
