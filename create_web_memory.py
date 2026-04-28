from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DB_FAISS_PATH = "vectorstore/db_faiss"

# 1. Put your specific PubMed or API India article URLs here
TARGET_URLS = [
    "https://pubmed.ncbi.nlm.nih.gov/29659364/", # Real article on Type 2 Diabetes
    "https://pubmed.ncbi.nlm.nih.gov/35771962/",
    "https://pubmed.ncbi.nlm.nih.gov/32741486/",
    "https://pubmed.ncbi.nlm.nih.gov/21193628/",
    "https://pubmed.ncbi.nlm.nih.gov/35146976/",
    "https://pubmed.ncbi.nlm.nih.gov/34238458/",
    "https://pubmed.ncbi.nlm.nih.gov/34238462/",
    "https://pubmed.ncbi.nlm.nih.gov/25330715/",
    "https://pubmed.ncbi.nlm.nih.gov/32383239/",
    "https://pubmed.ncbi.nlm.nih.gov/36007536/",
    "https://pubmed.ncbi.nlm.nih.gov/34522399/",
    "https://pubmed.ncbi.nlm.nih.gov/35758526/",
    "https://pubmed.ncbi.nlm.nih.gov/23985879/",
    "https://pubmed.ncbi.nlm.nih.gov/32092238/",
    "https://pubmed.ncbi.nlm.nih.gov/38841840/",
    "https://pubmed.ncbi.nlm.nih.gov/25786418/",
    "https://pubmed.ncbi.nlm.nih.gov/29135402/",
    "https://pubmed.ncbi.nlm.nih.gov/39985010/",
    "https://pubmed.ncbi.nlm.nih.gov/34806762/",
    "https://pubmed.ncbi.nlm.nih.gov/35023459/",
    "https://pubmed.ncbi.nlm.nih.gov/8868038/",
    "https://pubmed.ncbi.nlm.nih.gov/33400058/",#COVID-19 related articles
    "https://pubmed.ncbi.nlm.nih.gov/33126180/",
    "https://pubmed.ncbi.nlm.nih.gov/33664170/",
    "https://pubmed.ncbi.nlm.nih.gov/32563999/",
    "https://pubmed.ncbi.nlm.nih.gov/32749914/",
    "https://pubmed.ncbi.nlm.nih.gov/33572857/",
    "https://pubmed.ncbi.nlm.nih.gov/33980687/",
    "https://pubmed.ncbi.nlm.nih.gov/32685022/",
    "https://pubmed.ncbi.nlm.nih.gov/32395792/",
    "https://pubmed.ncbi.nlm.nih.gov/32321823/",










]

def create_web_vector_db():
    print("Scraping websites...")
    # WebBaseLoader automatically grabs the text AND saves the URL in doc.metadata['source']
    loader = WebBaseLoader(TARGET_URLS)
    documents = loader.load()

    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    print("Creating Embeddings and saving to FAISS...")
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.from_documents(texts, embedding_model)
    
    # Save the new database
    db.save_local(DB_FAISS_PATH)
    print("Web-linked Database Created Successfully!")

if __name__ == "__main__":
    create_web_vector_db()