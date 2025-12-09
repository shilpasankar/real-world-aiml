import streamlit as st
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA


# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="TruthLens AI — AI Detective",
    page_icon="🕵️‍♂️",
    layout="wide"
)

st.title("🕵️‍♂️ TruthLens AI — Document Contradiction & Claim Detector")
st.write("Upload documents and let the AI Detective uncover contradictions, missing information, and suspicious claims.")


# ------------------- SIDEBAR CONTROLS -------------------
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


# ------------------- TEXT EXTRACTION -------------------
def extract_text(files):
    text = ""
    for file in files:
        if file.type == "application/pdf":
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        else:
            text += file.read().decode("utf-8") + "\n"
    return text


# ------------------- ANALYSIS CORE -------------------
def run_analysis(full_text, modes):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80
    )
    chunks = splitter.split_text(full_text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_texts(chunks, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 5})

    llm = OpenAI(temperature=0)

    prompt = f"""
    You are TruthLens AI — a forensic reasoning engine.

    Analyze contradictions, missing information, and suspicious claims
    across the retrieved excerpts.

    Requested modes: {', '.join(modes)}

    For each finding:
    - Give a short headline
    - Explain the issue clear & concisely
    - Include the exact evidence text
    - Provide a severity rating (Low, Medium, High)

    Return results in clean, readable markdown sections.
    """

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    return qa.run(prompt)


# ------------------- MAIN LOGIC -------------------
if run_button and uploaded_files:
    with st.spinner("Detective analyzing your documents… 🕵️‍♂️"):
        full_text = extract_text(uploaded_files)
        findings = run_analysis(full_text, analysis_modes)

    st.success("Analysis complete! Findings below 👇")

    tab1, tab2 = st.tabs(["🔍 Findings", "📄 Document Text"])

    with tab1:
        st.markdown("## 🔍 Findings")
        st.write(findings)

    with tab2:
        st.markdown("## 📄 Full Extracted Text")
        st.text(full_text)

else:
    st.info("Upload documents and click **Run Analysis** to begin.")
