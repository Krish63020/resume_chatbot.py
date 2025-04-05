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

st.title("Resume Chatbot")
st.sidebar.header("Upload Resumes")
uploaded_files = st.sidebar.file_uploader("Upload Resume PDFs", type="pdf", accept_multiple_files=True)

def ensure_model_pulled(model_name):
    """Ensure the specified Ollama model is pulled."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if model_name not in result.stdout:
            st.info(f"Model '{model_name}' not found. Pulling it now...")
            subprocess.run(["ollama", "pull", model_name], check=True)
            st.success(f"Model '{model_name}' pulled successfully!")
    except subprocess.CalledProcessError as e:
        st.error(f"Failed to pull model '{model_name}': {e}")
        raise
    except FileNotFoundError:
        st.error("Ollama is not installed or not found in PATH. Please install it.")
        raise

# Initialize LLM and embeddings
try:
    llm = Ollama(model="llama3")
    embedding_model = "mxbai-embed-large"
    ensure_model_pulled(embedding_model)
    embeddings = OllamaEmbeddings(model=embedding_model)
except Exception as e:
    st.error(f"Failed to initialize LLM or embeddings: {e}")
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
            # Use a temporary file to avoid naming conflicts
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
    
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

if uploaded_files and st.sidebar.button("Process Resumes"):
    with st.spinner("Processing resumes..."):
        process_resumes(uploaded_files)

# Question input and response
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

# Reset button
if st.sidebar.button("Clear All"):
    st.session_state.vectorstore = None
    st.session_state.qa_chain = None
    st.success("Chatbot reset!")
