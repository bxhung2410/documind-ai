import os
import time # Import the time module
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

# --- 1. DEFINE THE PERSISTENT DIRECTORY ---
PERSIST_DIRECTORY = 'db'

def main():
    """
    Main function to run the document processing and QA.
    """
    print("--- Starting the AI Document Assistant (using Google Gemini) ---")

    # --- 2. LOAD THE DOCUMENT ---
    print("Loading document...")
    loader = PyPDFLoader("docs/impactReport.pdf")
    documents = loader.load()
    if not documents:
        print("Could not load the document. Please check the file path.")
        return
    print(f"Document loaded successfully. It has {len(documents)} page(s).")

    # --- 3. SPLIT THE DOCUMENT INTO CHUNKS ---
    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"Document split into {len(texts)} chunks.")

    # --- 4. CREATE EMBEDDINGS AND VECTOR STORE (WITH BATCHING) ---
    print("Creating embeddings and vector store with Google's model...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # --- MODIFIED LOGIC FOR BATCH PROCESSING ---
    # Create an empty Chroma vector store
    vectordb = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )
    
    # Add documents in batches to respect rate limits
    batch_size = 50
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"Adding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size} to vector store...")
        vectordb.add_documents(documents=batch)
        time.sleep(1) # Wait for 1 second between batches

    vectordb.persist()
    print("Vector store created and saved to disk.")

    # --- 5. CREATE THE QUESTION-ANSWERING CHAIN ---
    print("Creating the QA chain with Gemini Pro...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0, convert_system_message_to_human=True)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever()
    )
    print("QA chain created successfully.")

    # --- 6. ASK A QUESTION ---
    print("--- Ready to answer questions ---")
    question = "What is the main topic of this document?"
    
    try:
        print(f"\nAsking: {question}")
        result = qa_chain.invoke(question)
        print("\nAnswer:")
        print(result['result'])
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()