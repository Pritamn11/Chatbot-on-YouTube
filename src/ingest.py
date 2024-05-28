from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import SentenceTransformerEmbeddings 
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
import os


persist_directory = 'db'

def main():
    documents=[]
    for root, dirs, files in os.walk('docs'):
        for file in files:
            if file.endswith(".txt"):
                print(file)
                loader = TextLoader(os.path.join(root,file))
                documents.extend(loader.load())
    # documents = loader.load()
    if not documents:
        print("No documents found to process.")
        return
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create vector store
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
    
    db.persist()
    db=None


if __name__=="__main__":
    main()

