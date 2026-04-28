import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI

# ==========================================
# STEP 1: SETUP NVIDIA API
# ==========================================
# Yahan par apni NVIDIA wali key daalein (jo nvapi- se shuru hoti hai)
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
client = OpenAI(
  base_url="https://integrate.api.nvidia.com/v1",
  api_key=NVIDIA_API_KEY
)

# ==========================================
# STEP 2: LOAD FAISS DATABASE (Yeh apka sahi chal raha hai)
# ==========================================
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={'k': 3})

# ==========================================
# STEP 3: THE CHATBOT LOOP
# ==========================================
while True:
    user_query = input("Write Query Here (or type 'exit' to quit): ")
    
    if user_query.lower() == 'exit':
        break

    print("Searching medical database...")
    # Database se data nikalna
    docs = retriever.invoke(user_query)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Prompt banana
    final_prompt = f"""
    Use the pieces of information provided in the context to answer user's question.
    If you dont know the answer, just say that you dont know, dont try to make up an answer. 
    Dont provide anything out of the given context

    Context: {context}
    Question: {user_query}

    Start the answer directly. No small talk please.
    """

    print("Generating answer via NVIDIA...\n")
    
    # Seedha NVIDIA ko request bhejna
    response = client.chat.completions.create(
        model="mistralai/mistral-large-3-675b-instruct-2512", # Aapke screenshot wala premium model!
        messages=[{"role": "user", "content": final_prompt}],
        temperature=0.5,
        max_tokens=512
    )

    print("RESULT: \n", response.choices[0].message.content)
    print("-" * 50)