# app.py
import streamlit as st
import faiss
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import PyPDF2
from gtts import gTTS
import tempfile
import os

st.set_page_config(page_title="DocuRAG + TTS (fixed)", layout="wide")
st.title("📚 DocuRAG (HuggingFace) + TTS — fixed audio/session behavior")

# -----------------------------
# Model loading (cached)
# -----------------------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    return embedder, summarizer, qa_pipeline

embedder, summarizer, qa_pipeline = load_models()

# -----------------------------
# Session state initialization
# -----------------------------
if "text_chunks" not in st.session_state:
    st.session_state["text_chunks"] = None
if "index" not in st.session_state:
    st.session_state["index"] = None
if "summary" not in st.session_state:
    st.session_state["summary"] = None
if "summary_audio" not in st.session_state:
    st.session_state["summary_audio"] = None
if "answer" not in st.session_state:
    st.session_state["answer"] = None
if "answer_audio" not in st.session_state:
    st.session_state["answer_audio"] = None
if "uploaded_name" not in st.session_state:
    st.session_state["uploaded_name"] = None

# -----------------------------
# Helpers
# -----------------------------
def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def chunk_text_for_embedding(text, chunk_size=500, overlap=50):
    text = text.replace("\n", " ")
    chunks = []
    i = 0
    while i < len(text):
        end = min(len(text), i + chunk_size)
        chunks.append(text[i:end])
        i = end - overlap if end - overlap > i else end
    return chunks

def build_vectorstore(text_chunks):
    embs = embedder.encode(text_chunks, convert_to_numpy=True).astype("float32")
    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embs))
    return index, embs

def retrieve_top_k(query, index, text_chunks, top_k=3):
    q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(np.array(q_emb), top_k)
    results = [text_chunks[i] for i in I[0] if i != -1]
    return results

def summarize_long_text(text, summarizer, chunk_size=1000):
    if len(text) < chunk_size:
        s = summarizer(text, max_length=150, min_length=30, do_sample=False)[0]["summary_text"]
        return s
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    partials = []
    for c in chunks:
        try:
            part = summarizer(c, max_length=120, min_length=20, do_sample=False)[0]["summary_text"]
        except Exception:
            part = c[:200]
        partials.append(part)
    combined = " ".join(partials)
    final = summarizer(combined, max_length=180, min_length=40, do_sample=False)[0]["summary_text"]
    return final

def text_to_speech_bytes(text, lang="en"):
    """
    Produce MP3 bytes using gTTS and return bytes.
    We write to a temporary file, read bytes, then remove file.
    """
    tts = gTTS(text=text, lang=lang)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp.close()
    try:
        tts.save(tmp.name)
        with open(tmp.name, "rb") as f:
            data = f.read()
    finally:
        try:
            os.remove(tmp.name)
        except Exception:
            pass
    return data

# -----------------------------
# UI
# -----------------------------
st.sidebar.header("TTS & Retriever settings")
tts_lang = st.sidebar.selectbox("TTS language", ["en", "hi", "fr", "es"], index=0)
top_k = st.sidebar.slider("Retriever: top-k chunks", 1, 8, 3)

uploaded_file = st.file_uploader("📂 Upload PDF or TXT", type=["pdf", "txt"])
if uploaded_file:
    # Only rebuild index if a new file is uploaded
    if st.session_state["uploaded_name"] != uploaded_file.name:
        st.session_state["summary"] = None
        st.session_state["summary_audio"] = None
        st.session_state["answer"] = None
        st.session_state["answer_audio"] = None
        st.session_state["uploaded_name"] = uploaded_file.name

        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        else:
            try:
                text = uploaded_file.read().decode("utf-8")
            except Exception:
                text = str(uploaded_file.read())

        if not text.strip():
            st.error("No text found in the uploaded file.")
        else:
            st.success("Document loaded.")
            st.markdown("### 🔎 Document preview (first 2000 chars)")
            st.write(text[:2000] + ("..." if len(text) > 2000 else ""))

            # build chunks & index and save to session state
            text_chunks = chunk_text_for_embedding(text)
            if len(text_chunks) == 0:
                st.error("No chunks created from the document.")
            else:
                index, embeddings = build_vectorstore(text_chunks)
                st.session_state["text_chunks"] = text_chunks
                st.session_state["index"] = index
                st.info(f"Indexed {len(text_chunks)} chunks.")

# Summarize
if st.button("Summarize Document"):
    if not st.session_state["text_chunks"]:
        st.error("No indexed document available. Upload a PDF or TXT first.")
    else:
        with st.spinner("Summarizing..."):
            # join full text from chunks (or keep original text in your design)
            full_text = " ".join(st.session_state["text_chunks"])
            summary = summarize_long_text(full_text, summarizer)
            st.session_state["summary"] = summary
            try:
                st.session_state["summary_audio"] = text_to_speech_bytes(summary, lang=tts_lang)
            except Exception as e:
                st.session_state["summary_audio"] = None
                st.error(f"TTS generation failed: {e}")

# Display summary if present in session state
if st.session_state["summary"]:
    st.subheader("📌 Summary")
    st.write(st.session_state["summary"])
    cols = st.columns([1, 1, 2])
    with cols[0]:
        if st.button("🔊 Play summary audio"):
            if st.session_state["summary_audio"]:
                st.audio(st.session_state["summary_audio"], format="audio/mp3")
            else:
                st.error("No audio available for summary.")
    with cols[1]:
        if st.session_state["summary_audio"]:
            st.download_button("Download summary audio", st.session_state["summary_audio"],
                               file_name="summary.mp3", mime="audio/mp3")

# Q&A
query = st.text_input("Ask a question from the document:")
if query:
    if not st.session_state["index"] or not st.session_state["text_chunks"]:
        st.error("No indexed document available. Upload a PDF or TXT first.")
    else:
        with st.spinner("Retrieving context..."):
            retrieved = retrieve_top_k(query, st.session_state["index"], st.session_state["text_chunks"], top_k=top_k)
            context = " ".join(retrieved)
        st.markdown("**Retrieved context (top-k):**")
        with st.expander("Show retrieved context"):
            for i, r in enumerate(retrieved, start=1):
                st.write(f" chunk {i}:")
                st.write(r[:800] + ("..." if len(r) > 800 else ""))

        with st.spinner("Answering..."):
            try:
                answer = qa_pipeline(question=query, context=context)["answer"]
            except Exception as e:
                answer = "Error from QA model: " + str(e)
            st.session_state["answer"] = answer
            try:
                st.session_state["answer_audio"] = text_to_speech_bytes(answer, lang=tts_lang)
            except Exception:
                st.session_state["answer_audio"] = None

# Display answer if present
if st.session_state["answer"]:
    st.subheader("🤖 Answer")
    st.write(st.session_state["answer"])
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("🔊 Play answer audio"):
            if st.session_state["answer_audio"]:
                st.audio(st.session_state["answer_audio"], format="audio/mp3")
            else:
                st.error("No audio available for answer.")
    with c2:
        if st.session_state["answer_audio"]:
            st.download_button("Download answer audio", st.session_state["answer_audio"],
                               file_name="answer.mp3", mime="audio/mp3")
