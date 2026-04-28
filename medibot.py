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
    # 1. 🎨 THEME-AWARE CSS (No hardcoded colors!)
    st.markdown("""
        <style>
        /* Smooth Sidebar Divider */
        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.2);
        }
        
        /* Rounded Chat Input Box */
        [data-testid="stChatInput"] {
            border-radius: 20px !important;
            border: 1px solid rgba(128, 128, 128, 0.3) !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # 2. 🌟 NATIVE HEADER (Looks good in both Dark & Light mode)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center; color: #3b82f6;'>MediBot 🩺</h1>", unsafe_allow_html=True)
    
    st.markdown("<p style='text-align: center; opacity: 0.8; font-size: 1.1em;'>Your AI Specialist for Diabetes, Hypertension & COVID-19</p>", unsafe_allow_html=True)
    st.divider()

    # 3. 📱 SIDEBAR STYLING
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100)
        st.header("💡 Ask Me About")
        
        st.markdown("""
        - 🩸 **Diabetes:** Diet, Insulin, Sugar Spikes
        - 🫀 **Hypertension:** BP Management, Stress
        - 🦠 **COVID-19:** Recovery, Symptoms, Immunity
        """)
        
        st.divider()
        st.info("📚 Trained strictly on verified PubMed clinical research.")
        st.caption("Developed for educational & research purposes.")

    # ... YAHAN SE BAAKI KA CHAT HISTORY WALA CODE SAME RAHEGA ...

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        # Add a welcoming first message from the bot
        st.session_state.messages.append({
            'role': 'assistant', 
            'content': "Hello! I am MediBot. I specialize in answering questions about **Diabetes, Hypertension, and COVID-19** based on clinical research. How can I help you today?"
        })

    # Display previous chat messages with AVATARS
    for message in st.session_state.messages:
        # User gets a person icon, Bot gets a stethoscope/robot icon
        avatar_icon = "👤" if message['role'] == 'user' else "🩺"
        st.chat_message(message['role'], avatar=avatar_icon).markdown(message['content'])

    # The Chat Input Box
    prompt = st.chat_input("Ask a medical question...")

    if prompt:
        # Show user message instantly with Avatar
        st.chat_message('user', avatar="👤").markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        try: 
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the database.")
                return

            # Use a custom spinner
            with st.spinner("🤖 MediBot is scanning clinical records..."):
                
                retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
                docs = retriever.invoke(prompt)
                
                context = ""
                sources = set() 
                
                for doc in docs:
                    context += doc.page_content + "\n\n"
                    url = str(doc.metadata.get('source', 'Unknown Source')).strip()
                    sources.add(url)

                final_prompt = f"""
                You are an expert Medical AI assistant strictly specialized in Diabetes, Hypertension, and COVID-19.
                Use the following retrieved context to answer the user's question.
                
                CRITICAL RULES:
                1. If the user asks about a disease other than Diabetes, Hypertension, or COVID-19, politely say: "I am a specialized bot trained only on Diabetes, Hypertension, and COVID-19. I cannot answer queries about other conditions."
                2. Do not mix up the treatments or symptoms of the three diseases.
                3. If the answer is not in the context, say you don't know. Do not guess.

                Context: {context}
                Question: {prompt}

                Answer directly, professionally, and provide structured points if necessary.
                """

                response = client.chat.completions.create(
                    model="meta/llama-3.1-8b-instruct", 
                    messages=[{"role": "user", "content": final_prompt}],
                    temperature=0.0
                )

                result_text = response.choices[0].message.content

                # Styled Links Section
                link_section = "\n\n---\n**🔗 Verified Sources:**\n"
                for link in sources:
                    if link != 'Unknown Source':
                        link_section += f"- [View Original Research on PubMed]({link})\n"

                final_output = result_text + link_section

            # Final Answer UI with Avatar
            st.chat_message('assistant', avatar="🩺").markdown(final_output)
            st.session_state.messages.append({'role': 'assistant', 'content': final_output})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()