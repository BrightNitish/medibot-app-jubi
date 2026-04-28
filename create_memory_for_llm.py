import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

def create_vector_db():
    # 1. Load Documents
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()

    # --- PRO TIP: Metadata Injection ---
    # Agar aapki PDF ka naam 'pubmed_article_1.pdf' hai, 
    # toh hum manually metadata mein uska source link daal sakte hain.
    for doc in documents:
        # Aap yahan logic laga sakte ho: agar filename mein 'pubmed' hai toh PubMed ka link do
        doc.metadata["source_url"] = "https://pubmed.ncbi.nlm.nih.gov/" 

    # 2. Split Text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # 3. Create Embeddings
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    # 4. Save to FAISS (Metadata will be saved automatically)
    db = FAISS.from_documents(texts, embedding_model)
    db.save_local(DB_FAISS_PATH)
    print("Database Created with Metadata!")

if __name__ == "__main__":
    create_vector_db()