# 🩺 MediBot: Specialized Medical AI Assistant

**Live Application:** [👉 Click here to test MediBot on Railway](https://medibot-app-jubi-production.up.railway.app/)

MediBot is an AI-powered conversational agent specifically designed to answer clinical and health-related queries regarding **Diabetes, Hypertension, and COVID-19**. It leverages an advanced Retrieval-Augmented Generation (RAG) architecture grounded strictly in verified clinical research scraped directly from PubMed.

## 🎯 Scope and Intent
The primary intent of this project is to create an AI assistant that **does not hallucinate** medical facts. 
* **Strictly Scoped:** The bot is engineered to strictly deny answering queries outside its 3 core diseases (Diabetes, Hypertension, COVID-19). 
* **Evidence-Based:** It does not rely on the LLM's internal pre-trained memory. Instead, it retrieves specific chunks from 30+ verified PubMed articles and synthesizes answers based *only* on that context.
* **Traceable:** Every response provides clickable citations to the exact original research papers used to formulate the answer.

## 🧠 Technical Architecture

![MediBot Architecture](architecture.jpg)

1. **Ingestion Phase:** Automates the fetching of specific Medical URLs using `WebBaseLoader` and chunks the text securely.
2. **Vectorization:** Converts medical text into semantic vectors using HuggingFace's lightweight `sentence-transformers/all-MiniLM-L6-v2` and stores them in a local **FAISS** database.
3. **Retrieval & LLM Generation:** Uses semantic search to find the top-k relevant chunks. These chunks are fed into **LLaMA-3.1-8B** (via NVIDIA API) alongside a strict Anti-Hallucination system prompt to generate the final response.

## 🛠️ Tech Stack
* **Frontend:** Streamlit (Theme-aware Light/Dark UI)
* **AI Framework:** LangChain
* **LLM:** LLaMA-3.1-8b-Instruct (via API)
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **Vector Database:** FAISS (CPU-optimized)
* **Deployment Cloud:** Railway

---

## 🚀 Local Setup & Installation

Follow these steps to run the MediBot project on your local Windows machine:

### Step 1: Clone the repository
```bash
git clone [https://github.com/BrightNitish/medibot-app-jubi.git](https://github.com/BrightNitish/medibot-app-jubi.git)
cd medibot-app-jubi
```

### Step 2: Create and Activate a Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Setup Environment Variables
Create a `.env` file in the root directory and add your NVIDIA API key:
```env
NVIDIA_API_KEY=nvapi-your_actual_api_key_here
```

### Step 5: Build the Vector Database (Knowledge Base)
Run the script to build the FAISS index locally:
```bash
python create_web_memory.py
```

### Step 6: Run the Streamlit Application
```bash
streamlit run medibot.py
```

---

## 👨‍💻 About the Developer
Developed by **Nitish Kumar**, a final-year B.Tech student specializing in Artificial Intelligence and Machine Learning at BIT Mesra. With a strong focus on Generative AI, RAG architectures, and Computer Vision (including medical image segmentation using YOLO), this project showcases applied AI engineering for healthcare solutions. 

* 🐙 **GitHub:** [github.com/BrightNitish](https://github.com/BrightNitish)

---
*⚠️ **Disclaimer:** MediBot is a research and educational project showcasing applied AI/ML engineering. It is not a replacement for professional medical advice, diagnosis, or treatment.*