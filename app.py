import os
import streamlit as st
import fitz  # PyMuPDF
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load Groq API Key from environment variable
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Global variables
index = None
doc_chunks = []

# Extract text from uploaded PDF
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

# Chunk large text into smaller pieces
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Create vector database using FAISS
def create_vector_db(chunks):
    embeddings = embed_model.encode(chunks)
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(np.array(embeddings))
    return faiss_index, embeddings

# Search the vector DB for relevant chunks
def search_query(query, index, chunks, k=3):
    query_vec = embed_model.encode([query])
    D, I = index.search(np.array(query_vec), k)
    return [chunks[i] for i in I[0]]

# Send query to Groq model
def query_llm(prompt):
    response = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="llama-3-70b-8192"
    )
    return response.choices[0].message.content

# Streamlit UI
st.set_page_config(page_title="📄 RAG PDF QA Bot", page_icon="📘")
st.title("📄 RAG-based PDF QA Bot")
st.markdown("Upload a PDF, then ask questions based on its content.")

# Upload PDF
pdf_file = st.file_uploader("Upload your PDF file", type="pdf")

if pdf_file:
    with st.spinner("🔍 Extracting and chunking PDF..."):
        raw_text = extract_text_from_pdf(pdf_file)
        doc_chunks = chunk_text(raw_text)
        index, _ = create_vector_db(doc_chunks)
    st.success("✅ PDF processed!")

# Ask question
if index:
    user_query = st.text_input("Ask your question here:")
    if st.button("Get Answer"):
        with st.spinner("🤖 Thinking..."):
            relevant_chunks = search_query(user_query, index, doc_chunks)
            context = "\n\n".join(relevant_chunks)
            prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {user_query}"
            answer = query_llm(prompt)
            st.markdown("### ✅ Answer:")
            st.markdown(answer)
