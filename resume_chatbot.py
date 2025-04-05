import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain.chains import RetrievalQA
import os
import subprocess
import tempfile
import requests  # Ensure this is installed

st.title("Resume Chatbot")
st.sidebar.header("Upload Resumes")
uploaded_files = st.sidebar.file_uploader("Upload Resume PDFs", type="pdf", accept_multiple_files=True)

# Check if Ollama is running
def is_ollama_running():
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=2)
        return response.status_code == 200
    except:
        return False

if not is_ollama_running():
    st.error("Ollama is not running. Please start Ollama and refresh this page.")
    st.stop()

# Function to check if a model exists in Ollama
def check_model_exists(model_name):
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        return model_name in result.stdout
    except:
        return False

# Initialize LLM and embeddings
try:
    llm_model = "llama3"
    embedding_model = "nomic-embed-text"

    # Check if models exist
    if not check_model_exists(llm_model):
        st.warning(f"Model '{llm_model}' not found. It will be pulled when first used.")

    if not check_model_exists(embedding_model):
        st.warning(f"Embedding model '{embedding_model}' not found. It will be pulled when first used.")

    llm = Ollama(model=llm_model)
    embeddings = OllamaEmbeddings(model=embedding_model)

except Exception as e:
    st.error(f"Failed to initialize models: {e}")
    st.stop()

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

def process_resumes(uploaded_files):
    """Process uploaded resume PDFs and create a QA chain."""
    documents = []
    temp_files = []

    try:
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getbuffer())
                temp_files.append(tmp.name)

                loader = PyPDFLoader(tmp.name)
                pages = loader.load()
                documents.extend(pages)

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # Create vector store
        with st.spinner("Creating vector embeddings..."):
            vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
            return_source_documents=False
        )

        # Store in session state
        st.session_state.vectorstore = vectorstore
        st.session_state.qa_chain = qa_chain
        st.success("Resumes processed successfully!")

    except Exception as e:
        st.error(f"Error processing resumes: {e}")

    finally:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

if uploaded_files and st.sidebar.button("Process Resumes"):
    with st.spinner("Processing resumes..."):
        process_resumes(uploaded_files)

# Ask questions
st.subheader("Ask a Question About the Resumes")
query = st.text_input("Enter your question:")

if query and st.session_state.qa_chain:
    with st.spinner("Generating answer..."):
        try:
            result = st.session_state.qa_chain({"query": query})
            st.write("**Answer:**", result["result"])
        except Exception as e:
            st.error(f"Error generating answer: {e}")
elif query:
    st.warning("Please upload and process resumes first.")

# Reset
if st.sidebar.button("Clear All"):
    st.session_state.vectorstore = None
    st.session_state.qa_chain = None
    st.success("Chatbot reset!")
