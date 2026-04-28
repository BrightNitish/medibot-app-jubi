import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI

# ==========================================
# 1. SETUP NVIDIA API CLIENT
# ==========================================
# Quotes hata diye hain, ab yeh sach mein .env ya environment se key uthayega
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY") 

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY 
)

DB_FAISS_PATH = "vectorstore/db_faiss"

# ==========================================
# 2. LOAD DATABASE (Cached)
# ==========================================
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# ==========================================
# 3. STREAMLIT UI & CHAT LOGIC
# ==========================================
def main():
    st.title("MediBot 🩺")
    st.caption("Powered by PubMed/API India & NVIDIA NIM")

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display previous chat messages
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # The Chat Input Box
    prompt = st.chat_input("Ask a medical question...")

    # YAHAN SE INDENTATION SAHI KI GAYI HAI
    if prompt:
        # Show user message instantly
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})

        try: 
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the database.")
                return

            with st.spinner("Searching PubMed & API India records..."):
                
                # FIX: Retriever ko define karna zaruri tha
                retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
                docs = retriever.invoke(prompt)
                
                # Context aur Unique Links collect karein
                context = ""
                sources = set() 
                
                for doc in docs:
                    context += doc.page_content + "\n\n"
                    
                    # FIX: str() aur .strip() lagaya taaki koi hidden space na rahe
                    raw_url = doc.metadata.get('source', 'Unknown Source')
                    url = str(raw_url).strip() 
                    
                    sources.add(url)

                # NVIDIA API Prompt
                final_prompt = f"Context: {context}\nQuestion: {prompt}\nAnswer directly based on context."

                response = client.chat.completions.create(
                    model="meta/llama-3.1-8b-instruct", 
                    messages=[{"role": "user", "content": final_prompt}],
                    temperature=0.0
                )

                result_text = response.choices[0].message.content

                # Clickable Links Section
                link_section = "\n\n**🔗 Verified Sources:**\n"
                for link in sources:
                    if link != 'Unknown Source':
                        link_section += f"- [View Original Research/Article]({link})\n"

                final_output = result_text + link_section

            # Final Answer UI mein dikhayein
            st.chat_message('assistant').markdown(final_output)
            st.session_state.messages.append({'role':'assistant', 'content': final_output})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()