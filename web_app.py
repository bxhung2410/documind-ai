import os
import time
import streamlit as st
from dotenv import load_dotenv

# LangChain components
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# Load environment variables from .env file
load_dotenv()

# --- Functions from previous script ---

def process_documents(document_path, db_directory):
    """
    Loads a single document, splits it, and creates a persisted vector store.
    """
    print(f"Loading document: {document_path}")
    loader = PyPDFLoader(document_path)
    documents = loader.load()

    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    print("Creating embeddings and vector store...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create a new vector store from the documents
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=db_directory
    )
    vectordb.persist()
    print("Vector store created and saved.")
    return vectordb

def get_qa_chain(db_directory):
    """
    Loads an existing vector store and creates a QA chain.
    """
    print("Loading existing vector store...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = Chroma(persist_directory=db_directory, embedding_function=embeddings)

    print("Creating the QA chain...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0, convert_system_message_to_human=True)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever()
    )
    return qa_chain

# --- Streamlit Web App Interface ---

# Set page title
st.set_page_config(page_title="Trợ lý Tài liệu AI")
st.title("💬 Trợ lý Tài liệu AI")

# Sidebar for file upload
with st.sidebar:
    st.header("Tài liệu của bạn")
    uploaded_file = st.file_uploader("Tải lên một file PDF và nhấn 'Xử lý'", type="pdf")
    
    if st.button("Xử lý"):
        if uploaded_file is not None:
            # Create a temporary directory for this session's data
            session_db_dir = f"db/{uploaded_file.file_id}"
            if not os.path.exists(session_db_dir):
                os.makedirs(session_db_dir)

            # Save the uploaded file temporarily
            filepath = os.path.join(session_db_dir, uploaded_file.name)
            with open(filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Đang xử lý tài liệu... (việc này có thể mất vài phút)"):
                process_documents(filepath, session_db_dir)
            
            st.success("Tài liệu đã được xử lý! Bây giờ bạn có thể hỏi.")
            st.session_state.db_dir = session_db_dir
            st.session_state.messages = [] # Clear previous messages
        else:
            st.warning("Vui lòng tải lên một file PDF.")

# Initialize or load chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main chat interface
if "db_dir" in st.session_state:
    # Create the QA chain from the processed document's database
    qa_chain = get_qa_chain(st.session_state.db_dir)

    if prompt := st.chat_input("Hãy hỏi một câu về tài liệu của bạn..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("AI đang suy nghĩ..."):
                try:
                    result = qa_chain.invoke(prompt)
                    response = result['result']
                    st.markdown(response)
                    # Add AI response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Đã xảy ra lỗi: {e}")
else:
    st.info("Chào mừng! Vui lòng tải lên một tài liệu PDF ở thanh bên trái để bắt đầu.")