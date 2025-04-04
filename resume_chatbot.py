import streamlit as st
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain.chains import RetrievalQA
import os
import subprocess

st.title("Resume Chatbot")
st.sidebar.header("Upload Resumes")
uploaded_files = st.sidebar.file_uploader("Upload Resume PDFs", type="pdf", accept_multiple_files=True)

def ensure_model_pulled(model_name):
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if model_name not in result.stdout:
            st.info(f"Model '{model_name}' not found. Pulling it now...")
            subprocess.run(["ollama", "pull", model_name], check=True)
            st.success(f"Model '{model_name}' pulled successfully!")
    except subprocess.CalledProcessError as e:
        st.error(f"Failed to pull model '{model_name}': {e}")
        raise

llm = Ollama(model="llama3")
embedding_model = "mxbai-embed-large"
ensure_model_pulled(embedding_model)
embeddings = OllamaEmbeddings(model=embedding_model)

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

def process_resumes(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        loader = PyPDFLoader(uploaded_file.name)
        pages = loader.load()
        documents.extend(pages)
        os.remove(uploaded_file.name)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=False
    )

    st.session_state.vectorstore = vectorstore
    st.session_state.qa_chain = qa_chain
    st.success("Resumes processed successfully!")

if uploaded_files and st.sidebar.button("Process Resumes"):
    with st.spinner("Processing resumes..."):
        process_resumes(uploaded_files)

st.subheader("Ask a Question About the Resumes")
query = st.text_input("Enter your question:")

if query and st.session_state.qa_chain:
    with st.spinner("Generating answer..."):
        result = st.session_state.qa_chain({"query": query})
        st.write("**Answer:**", result["result"])
elif query:
    st.warning("Please upload and process resumes first.")

if st.sidebar.button("Clear All"):
    st.session_state.vectorstore = None
    st.session_state.qa_chain = None
    st.success("Chatbot reset!")
