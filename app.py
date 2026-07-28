import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

st.set_page_config(page_title="Agente OCI", page_icon="🤖")
st.title("🤖 Agente Oracle Cloud")

if "vectorstore" not in st.session_state:
    with st.spinner("📚 Cargando documento..."):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        st.session_state.vectorstore = vectorstore
        st.session_state.client = genai.Client(
            api_key=os.getenv("GOOGLE_API_KEY")
        )
    st.success("✅ ¡Listo! Pregunta lo que necesites sobre Oracle Cloud.")

pregunta = st.text_input("Haz una pregunta sobre el PDF")

if st.button("Preguntar") and pregunta:
    with st.spinner("🤔 Pensando..."):
        docs = st.session_state.vectorstore.similarity_search(pregunta, k=4)
        contexto = "\n\n".join(doc.page_content for doc in docs)
        prompt = f"""
Responde SOLO utilizando la informacion entregada.

Contexto:
{contexto}

Pregunta:
{pregunta}
"""
        respuesta = st.session_state.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
    st.success(respuesta.text)