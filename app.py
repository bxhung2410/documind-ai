import os
import time
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

PERSIST_DIRECTORY = 'db'
DOCUMENTS_PATH = 'docs'

def process_documents():
    """
    Loads documents from the docs folder, splits them, and creates a persisted vector store.
    """
    print(f"Loading documents from {DOCUMENTS_PATH}...")
    # For this example, we'll just load the first PDF found.
    # In a real app, you might want to loop through all files.
    pdf_files = [f for f in os.listdir(DOCUMENTS_PATH) if f.endswith('.pdf')]
    if not pdf_files:
        print("No PDF files found in the 'docs' folder. Please add a PDF to continue.")
        return

    document_path = os.path.join(DOCUMENTS_PATH, pdf_files[0])
    loader = PyPDFLoader(document_path)
    documents = loader.load()
    print(f"Document '{pdf_files[0]}' loaded.")

    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"Document split into {len(texts)} chunks.")

    print("Creating embeddings and vector store...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    vectordb.persist()
    print("Vector store created and saved to disk.")

def main():
    """
    Main function to run the interactive QA session.
    """
    # Check if the vector store already exists
    if not os.path.exists(PERSIST_DIRECTORY):
        print("No existing vector store found. Processing documents...")
        process_documents()
    
    print("Loading existing vector store...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    
    print("Creating the QA chain...")
    llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0, convert_system_message_to_human=True)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever()
    )
    
    print("\n--- AI Document Assistant is Ready ---")
    print("Ask a question about your document. Type 'exit' to quit.")
    
    # Interactive loop
    while True:
        question = input("\nAsk a question: ")
        
        if question.lower() == 'exit':
            print("Goodbye!")
            break
        
        if not question.strip():
            continue
            
        try:
            print("Thinking...")
            result = qa_chain.invoke(question)
            print("\nAnswer:")
            print(result['result'])
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == '__main__':
    main()