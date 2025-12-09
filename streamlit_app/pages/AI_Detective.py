import streamlit as st
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA


# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="TruthLens AI — AI Detective",
    page_icon="assets/truthlens_logo.svg",
    layout="wide"
)

st.title("🕵️‍♂️ TruthLens AI — Document Contradiction & Claim Detector")
st.write("Upload documents and let the AI Detective uncover contradictions, missing information, and suspicious claims.")


# ----------------------------------------------------------
# SIDEBAR SETTINGS
# ----------------------------------------------------------
st.sidebar.header("🔧 Settings")

uploaded_files = st.sidebar.file_uploader(
    "Upload multiple documents (PDF or TXT)",
    accept_multiple_files=True
)

analysis_modes = st.sidebar.multiselect(
    "Detective Modes",
    ["Contradictions", "Missing Information", "Suspicious Claims"],
    default=["Contradictions", "Missing Information"]
)

temperature = st.sidebar.slider("Model Creativity", 0.0, 1.0, 0.0)

run_button = st.sidebar.button("Run Analysis")

st.sidebar.markdown("---")
st.sidebar.caption("TruthLens AI © 2025")


# ----------------------------------------------------------
# TEXT EXTRACTION (PDF + TXT)
# ----------------------------------------------------------
def extract_text(files):
    """Extracts text from PDF or TXT files using pdfplumber."""
    text = ""

    for file in files:
        if file.type == "application/pdf":
            try:
                with pdfplumber.open(file) as pdf:
                    for page in pdf.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text += extracted + "\n"
            except Exception as e:
                st.error(f"❌ Error reading PDF: {e}")

        else:
            try:
                text += file.read().decode("utf-8") + "\n"
            except:
                st.error("❌ Unable to decode text file.")

    return text


# ----------------------------------------------------------
# ANALYSIS PIPELINE
# ----------------------------------------------------------
def run_analysis(full_text, modes):
    """Runs chunking → embeddings → retrieval → LLM reasoning."""
    
    # 1. Chunk text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80
    )
    chunks = splitter.split_text(full_text)

    # 2. Embed chunks
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.from_texts(chunks, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 5})

    # 3. LLM (OpenAI)
    llm = OpenAI(temperature=temperature)

    # 4. Prompt
    prompt = f"""
    You are TruthLens AI — a forensic reasoning engine.

    Analyze the retrieved excerpts for:
    {', '.join(modes)}

    For each issue:
    - Provide a short title
    - Explain the contradiction or missing information clearly
    - Include the exact evidence text
    - Rate severity: Low, Medium, High

    Respond in clean, structured markdown.
    """

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    return qa.run(prompt)


# ----------------------------------------------------------
# MAIN APP LOGIC
# ----------------------------------------------------------
if run_button and uploaded_files:
    with st.spinner("🕵️ Detective analyzing your documents…"):
        full_text = extract_text(uploaded_files)
        findings = run_analysis(full_text, analysis_modes)

    st.success("✅ Analysis complete! Findings below.")

    # Tabs
    tab1, tab2 = st.tabs(["🔍 Findings", "📄 Document Text"])

    with tab1:
        st.markdown("## 🔍 Findings")
        st.write(findings)

    with tab2:
        st.markdown("## 📄 Full Extracted Document Text")
        st.text(full_text)

else:
    st.info("Upload documents and click **Run Analysis** to begin.")
