import os 
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 
from pathlib import Path

load_dotenv()

ROOT_DIR = Path(__file__).parent.parent
DOCS_DIR = ROOT_DIR / "docs"
DB_DIR = ROOT_DIR / "db"

def load_pdfs(directory_path):
    print(f"Loading directory: {directory_path}")

    if not directory_path.exists():
        raise FileNotFoundError(f"Directory doesnot exist")
    
    print("Directory Found. Loading PDFs...")

    documents = []

    for filepath in directory_path.glob("*pdf"):
        print(f"Loading file: {filepath.name}")
        loader = PyPDFLoader(str(filepath))
        documents.extend(loader.load())
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)

    if chunks:
        print("-----------------------------------------------------")
        print(f"Total chunks created: {len(chunks)}")
    return chunks

def create_embeddings(chunks,database_path):
    # embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(database_path):
        print("Loading existing vector database...")
        vectorstore = Chroma(persist_directory=database_path, embedding_function=embeddings)
        print("Vector database loaded successfully")
    else:
        print("Creating a vector database...")
        vectorstore = Chroma.from_documents(chunks,embedding=embeddings,persist_directory=database_path)
        print("Finished creating vector database")
    
    return vectorstore


def main():
    print("Starting ingestion pipeline...")

    #load pdfs 
    documents = load_pdfs(DOCS_DIR)

    #split documents into chunks
    chunks = split_documents(documents)

    #create embeddings and store in vector database
    vectorstore = create_embeddings(chunks,DB_DIR)

    print("Ingestion pipeline completed successfully")
    return vectorstore

if __name__ == "__main__":
    main()