import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from agent.loader import cargar_documentos

# 1. Cargar PDF
documentos = cargar_documentos("data/OC-Infra-Funda_sg.pdf")

# 2. Crear embeddings locales (GRATIS, sin API key, sin cuota)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 3. Crear y guardar índice FAISS
vectorstore = FAISS.from_documents(documentos, embeddings)
vectorstore.save_local("faiss_index")

print("✅ Índice guardado en 'faiss_index/'")