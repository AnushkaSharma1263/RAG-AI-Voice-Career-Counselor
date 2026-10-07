# MinuteHire Voice Career Counselor (Streamlit + Groq)

Speak, it answers aloud. RAG-grounded career counseling. One secret: `GROQ_API_KEY`.

## Pipeline
```
st.audio_input (mic) -> Groq whisper-large-v3-turbo
  -> local bge-small embedding -> FAISS top-4 (33 curated chunks)
  -> Groq llama-3.3-70b-versatile (streamed)
  -> clause/sentence chunker (<=190 chars) -> Groq Orpheus TTS per chunk, parallel (4 threads)
  -> each clip pushed to a browser-side queue the moment it is ready (speech starts before the reply finishes)
```
Typed input (chat box) goes through the same pipeline. History (last 12 messages) lives in `st.session_state`.

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repo (root must contain `streamlit_app.py`).
2. share.streamlit.io -> New app -> pick repo, branch, main file `streamlit_app.py`.
3. Advanced settings -> Python 3.12 (or 3.11) and Secrets:
   ```
   GROQ_API_KEY = "gsk_..."
   ```
4. Deploy. Submit the `https://<name>.streamlit.app` URL.

First boot downloads the ~130 MB embedding model (about a minute). Community Cloud sleeps idle apps: open the URL once before the evaluator does.

## Run locally
```
pip install -r requirements.txt
streamlit run streamlit_app.py     # needs GROQ_API_KEY in env or .streamlit/secrets.toml
```

## Latency decisions
- LLM streamed; first TTS chunk cut at the first clause, later chunks at sentence ends.
- TTS for all chunks runs in parallel threads while the LLM keeps generating; clips play in order.
- Embedding model + FAISS index cached once per server (`st.cache_resource`), query embeddings cached.
- No query-rewrite LLM call: previous student turn is appended to the retrieval query.
- Short answers (~60 words). Per-turn timings shown under the chat.

## Known limits (be upfront in the interview)
- Streamlit reruns the script per turn and `st.audio_input` is tap-to-record/tap-to-stop, so there is no hands-free VAD or instant barge-in (a new turn does cut off old audio). The FastAPI/WebSocket version in the other zip has both.
- Browsers may block autoplay (notably Safari): use "Replay last answer".
- Orpheus is a Groq preview model; override with `TTS_MODEL` / `TTS_VOICE` env vars if it changes.
- Tune `MIN_SCORE` (default 0.45) if retrieval is too loose or strict.
