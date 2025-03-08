# FAISS-based Retrieval-Augmented Generation (RAG) with Gemini LLM

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using **FAISS** for document storage and **Google Gemini API** for question-answering.

## 🚀 Features
- **Vector Storage:** Uses FAISS to store and retrieve relevant documents.
- **Embedding Model:** Leverages HuggingFace `sentence-transformers/all-MiniLM-L6-v2` for generating text embeddings.
- **LLM Integration:** Queries **Google Gemini API** to generate responses based on retrieved context.
- **Efficient Retrieval:** FAISS ensures quick document search and retrieval.
- **API Key Security:** Stores sensitive credentials securely using `.env`.

## 📌 Installation
Ensure you have **Python 3.8+** installed, then run:

```bash
pip install --upgrade langchain langchain-google-genai google-generativeai faiss-cpu sentence-transformers python-dotenv
```

## 📂 Project Structure
```
├── faiss_index/              # FAISS stored index
├── faiss_docs.pkl            # Pickled documents for retrieval
├── .env                      # API key (not committed to Git)
├── app.py                    # Main application script
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## 🔑 API Key Configuration
1. Create a `.env` file in the project root and add:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```
2. Load it in `app.py`:
   ```python
   from dotenv import load_dotenv
   import os
   load_dotenv()
   genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
   ```

## 📖 Usage
### 1️⃣ **Indexing Documents**
```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import pickle

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Process documents
vector_store = FAISS.from_documents(chunks, embedding_model)
vector_store.save_local("faiss_index")
with open("faiss_docs.pkl", "wb") as f:
    pickle.dump(documents, f)
```

### 2️⃣ **Retrieval & Generation**
```python
# Load FAISS index
vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever()

# Query Gemini model
def query_gemini(query, context):
    model = genai.GenerativeModel("gemini-pro")
    prompt = f"Answer the following question based on the provided context:\n\nContext: {context}\n\nQuestion: {query}"
    response = model.generate_content(prompt)
    return response.text

query = "What is NLP?"
retrieved_docs = retriever.get_relevant_documents(query)
context = "\n".join([doc.page_content for doc in retrieved_docs])
response = query_gemini(query, context)
print(f"Q: {query}\nA: {response}")
```

## 🛠 Troubleshooting
- **API Key Not Found:** Ensure the `.env` file is correctly set up and re-run `load_dotenv()`.
- **FAISS Load Error:** Set `allow_dangerous_deserialization=True` when loading the FAISS index.
- **Embeddings Issue:** Ensure `sentence-transformers` is installed and properly referenced.

## 📜 License
This project is licensed under the **MIT License**.

## 📬 Contact
For queries or suggestions, feel free to reach out!

---
🎯 **Happy Coding!** 🚀

