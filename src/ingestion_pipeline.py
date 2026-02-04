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

def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    pass


def main():
    print("Starting ingestion pipeline...")

    #load pdfs 
    documents = load_pdfs(DOCS_DIR)

    #split documents into chunks
    


if __name__ == "__main__":
    main()