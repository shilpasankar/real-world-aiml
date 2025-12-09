import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
from pypdf import PdfReader
import numpy as np

st.title("🕵️ AI Detective – PDF Investigator")

# -----------------------
# Load Local Embedding Model
# -----------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedder = load_embedder()

# -----------------------
# Load Local LLM
# -----------------------
@st.cache_resource
def load_llm():
    # You can swap this for any local HF model you have
    return pipeline(
        "text-generation",
        model="tiiuae/falcon-7b-instruct",
        device_map="auto"
    )

llm = load_llm()

# -----------------------
# PDF → Text
# -----------------------
def pdf_to_text(uploaded_pdf):
    reader = PdfReader(uploaded_pdf)
    texts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        texts.append(page_text)
    return "\n".join(texts)

# -----------------------
# Chunking helper
# -----------------------
def chunk_text(text, chunk_size=800, overlap=200):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# -----------------------
# Build FAISS index
# -----------------------
def build_faiss_index(chunks):
    embeddings = embedder.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, chunks

# Keep state across interactions
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None

# -----------------------
# UI: PDF Upload
# -----------------------
uploaded_pdf = st.file_uploader("Upload a PDF to investigate", type=["pdf"])

if uploaded_pdf is not None:
    with st.spinner("Reading and indexing PDF..."):
        text = pdf_to_text(uploaded_pdf)
        if not text.strip():
            st.error("Could not extract text from this PDF.")
        else:
            chunks = chunk_text(text)
            index, chunks = build_faiss_index(chunks)
            st.session_state.faiss_index = index
            st.session_state.chunks = chunks
            st.success(f"Indexed {len(chunks)} chunks from the PDF ✅")

st.write("---")

# -----------------------
# Question Input
# -----------------------
query = st.text_input("Ask the AI Detective a question about this PDF:")

def search_chunks(query, k=5):
    q_emb = embedder.encode([query]).astype("float32")
    D, I = st.session_state.faiss_index.search(q_emb, k)
    indices = I[0]
    return [st.session_state.chunks[i] for i in indices if i < len(st.session_state.chunks)]

if query:
    if st.session_state.faiss_index is None:
        st.warning("Please upload a PDF first so I have something to investigate. 🕵️‍♂️")
    else:
        st.write("### 🔍 Relevant Excerpts:")
        retrieved = search_chunks(query, k=4)
        for i, chunk in enumerate(retrieved):
            with st.expander(f"Clue {i+1}"):
                st.write(chunk)

        st.write("---")
        context = "\n\n".join(retrieved)
        prompt = (
            "You are an assistant answering questions based ONLY on the context.\n"
            "Be concise and cite phrases from the context when relevant.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        st.write("### 🧠 Detective's Deduction:")
        with st.spinner("Thinking..."):
            out = llm(prompt, max_new_tokens=256, do_sample=True, temperature=0.3)[0]["generated_text"]
        # Optionally trim everything before "Answer:" if model echoes prompt
        if "Answer:" in out:
            out = out.split("Answer:", 1)[-1].strip()
        st.write(out)
