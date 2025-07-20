import os
import streamlit as st
import tempfile
import requests
from dotenv import load_dotenv
from langchain.document_loaders import CSVLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
 
# Load environment variables from .env
load_dotenv()
 
# Load Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not set in environment variables.")
 
# Gemini API endpoint
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent"
)
 
# Ensure embedding model is present
def ensure_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    try:
        SentenceTransformer(model_name)
        st.sidebar.info(f"✅ '{model_name}' embedding model is ready to use.")
    except Exception as e:
        st.sidebar.error(f"Error downloading embedding model: {str(e)}")
 
# Streamlit page setup
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("RAG-based Chatbot with Qdrant + Gemini Pro")
 
# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
 
# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
 
# File processing
def process_uploaded_file(uploaded_file):
    try:
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, uploaded_file.name)
 
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
 
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif uploaded_file.name.endswith(".csv"):
            loader = CSVLoader(file_path=file_path)
        else:
            raise ValueError("Unsupported file format (only PDF or CSV allowed)")
 
        return loader.load()
    except Exception as e:
        raise RuntimeError(f"File processing failed: {str(e)}")
 
# Setup RAG pipeline
def setup_rag_pipeline(documents):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
 
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
 
        # connect to local Qdrant docker
        qdrant_client = QdrantClient(url="http://localhost:6333")
 
        collection_name = "rag_chatbot_collection"
 
        vectorstore = Qdrant.from_documents(
            documents=splits,
            embedding=embeddings,
            url="http://localhost:6333",
            collection_name=collection_name
        )
 
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
 
        # store vectorstore and memory in session_state
        st.session_state.vectorstore = vectorstore
        st.session_state.memory = memory
 
        return True
    except Exception as e:
        raise RuntimeError(f"RAG pipeline setup failed: {str(e)}")
 
# Sidebar upload
with st.sidebar:
    st.header("Upload Knowledge Base")
    ensure_embedding_model()
    uploaded_file = st.file_uploader("Upload PDF or CSV", type=["pdf", "csv"])
 
    if uploaded_file:
        try:
            documents = process_uploaded_file(uploaded_file)
            setup_rag_pipeline(documents)
            st.success("✅ Knowledge base processed and indexed successfully!")
        except Exception as e:
            st.error(f"Error: {str(e)}")
 
# User prompt
if prompt := st.chat_input("Ask me anything about the knowledge base"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
 
    # check if pipeline loaded
    if "vectorstore" not in st.session_state or "memory" not in st.session_state:
        st.error("⚠️ Please upload a knowledge base file first.")
        st.stop()
 
    try:
        # retrieve top relevant chunks
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.get_relevant_documents(prompt)
 
        # build a context string for Gemini
        context = "\n".join([doc.page_content for doc in relevant_docs])
 
        # build the Gemini prompt
        gemini_prompt = f"""
You are a helpful RAG assistant. Use the following context to answer:
---
{context}
---
Question: {prompt}
Answer:
"""
 
        # Gemini API call
        response = requests.post(
            GEMINI_API_URL,
            params={"key": GEMINI_API_KEY},
            json={
                "contents": [
                    {
                        "parts": [{"text": gemini_prompt}]
                    }
                ]
            },
            timeout=30
        )
 
        if response.status_code != 200:
            raise RuntimeError(f"Gemini API error: {response.status_code} {response.text}")
 
        gemini_answer = (
            response.json()
            .get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "Sorry, I could not generate an answer.")
        )
 
        with st.chat_message("assistant"):
            st.markdown(gemini_answer)
 
        st.session_state.messages.append({"role": "assistant", "content": gemini_answer})
 
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
 
