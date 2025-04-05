import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain.chains import RetrievalQA
from langchain.schema import Document

import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

# --- UI Setup ---
st.title("Resume Chatbot")
st.sidebar.header("Upload Resumes")
uploaded_files = st.sidebar.file_uploader("Upload Resume PDFs", type="pdf", accept_multiple_files=True)

# --- Ollama Check ---
def is_ollama_running():
    try:
        import requests
        response = requests.get("http://localhost:11434/api/version", timeout=2)
        return response.status_code == 200
    except:
        return False

if not is_ollama_running():
    st.error("Ollama is not running. Please start Ollama and refresh this page.")
    st.stop()

# --- Model Check ---
def check_model_exists(model_name):
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        return model_name in result.stdout
    except:
        return False

try:
    llm_model = "llama3"
    embedding_model = "nomic-embed-text"

    if not check_model_exists(llm_model):
        st.warning(f"Model '{llm_model}' not found. Will attempt to pull when needed.")
    if not check_model_exists(embedding_model):
        st.warning(f"Embedding model '{embedding_model}' not found. Will attempt to pull when needed.")

    llm = Ollama(model=llm_model)
    embeddings = OllamaEmbeddings(model=embedding_model)

except Exception as e:
    st.error(f"Failed to initialize models: {e}")
    st.stop()

# --- Session State ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# --- Resume Processing Function ---
def load_pdf(path):
    loader = PyPDFLoader(path)
    return loader.load()

def process_resumes(uploaded_files):
    documents = []
    temp_files = []

    try:
        # Save uploaded PDFs as temp files
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getbuffer())
                temp_files.append(tmp.name)

        # Load PDFs in parallel
        with st.spinner("Reading and extracting text from PDFs..."):
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(load_pdf, temp_files))
                for pages in results:
                    documents.extend(pages)

        # Convert to plain text
        raw_texts = [doc.page_content for doc in documents]
        documents = [Document(page_content=txt) for txt in raw_texts]

        # Split text
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)

        # Embed documents
        with st.spinner("Creating vector embeddings... This might take a while."):
            vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

        # Build QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
            return_source_documents=False
        )

        # Save in session
        st.session_state.vectorstore = vectorstore
        st.session_state.qa_chain = qa_chain

        st.success("Resumes processed successfully!")

    except Exception as e:
        st.error(f"Error processing resumes: {e}")

    finally:
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)

# --- Run Resume Processing ---
if uploaded_files and st.sidebar.button("Process Resumes"):
    with st.spinner("Processing resumes..."):
        process_resumes(uploaded_files)

# --- Question Input and Answer Generation ---
st.subheader("Ask a Question About the Resumes")
query = st.text_input("Enter your question:")

if query:
    if st.session_state.qa_chain:
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.qa_chain.run(query)
                if result.strip():
                    st.markdown(f"**Answer:** {result}")
                else:
                    st.warning("No relevant information found in the resumes.")
            except Exception as e:
                st.error(f"Something went wrong while generating the answer: {e}")
    else:
        st.warning("Please upload and process resumes first.")

# --- Reset Button ---
if st.sidebar.button("Clear All"):
    st.session_state.vectorstore = None
    st.session_state.qa_chain = None
    st.success("Chatbot reset!")
