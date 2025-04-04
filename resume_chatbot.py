import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import os
import subprocess

# ---- Streamlit UI Setup ----
st.title("Resume Chatbot")
st.sidebar.header("Upload Resumes")

# ---- Upload Multiple Resume Files ----
uploaded_files = st.sidebar.file_uploader("Upload Resume PDFs", type="pdf", accept_multiple_files=True)

# ---- Function to Ensure Model is Pulled ----
def ensure_model_pulled(model_name):
    try:
        # Check if the model exists by listing models
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        print(result.stdout)
        if model_name not in result.stdout:
            st.info(f"Model '{model_name}' not found. Pulling it now...")
            subprocess.run(["ollama", "pull", model_name], check=True)
            st.success(f"Model '{model_name}' pulled successfully!")
    except subprocess.CalledProcessError as e:
        st.error(f"Failed to pull model '{model_name}': {e}")
        raise

# ---- Initialize Ollama Model and Embeddings ----
llm = Ollama(model="llama3.2")  # Fast and lightweight model
embedding_model = "mxbai-embed-large"
ensure_model_pulled(embedding_model)  # Ensure embedding model is available
embeddings = OllamaEmbeddings(model=embedding_model)

# ---- Session State for Persistence ----
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# ---- Process Uploaded Resumes ----
def process_resumes(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        # Save the uploaded file temporarily
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load PDF content
        loader = PyPDFLoader(uploaded_file.name)
        pages = loader.load()
        documents.extend(pages)

        # Clean up temporary file
        os.remove(uploaded_file.name)

    # Split documents into smaller chunks for faster processing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Small chunks for quick retrieval
        chunk_overlap=200  # Some overlap for context
    )
    chunks = text_splitter.split_documents(documents)

    # Create Chroma vector store (in-memory for speed)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # Set up RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # Retrieve top 3 relevant chunks
        ),
        return_source_documents=False
    )

    # Store in session state
    st.session_state.vectorstore = vectorstore
    st.session_state.qa_chain = qa_chain
    st.success("Resumes processed successfully!")

# ---- Process Resumes When Uploaded ----
if uploaded_files and st.sidebar.button("Process Resumes"):
    with st.spinner("Processing resumes..."):
        process_resumes(uploaded_files)

# ---- Chat Interface ----
st.subheader("Ask a Question About the Resumes")
query = st.text_input("Enter your question:")

if query and st.session_state.qa_chain:
    with st.spinner("Generating answer..."):
        result = st.session_state.qa_chain({"query": query})
        st.write("**Answer:**", result["result"])
elif query:
    st.warning("Please upload and process resumes first.")

# ---- Clear Session State (Optional) ----
if st.sidebar.button("Clear All"):
    st.session_state.vectorstore = None
    st.session_state.qa_chain = None
    st.success("Chatbot reset!")