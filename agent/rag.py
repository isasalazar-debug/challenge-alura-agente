import os
from dotenv import load_dotenv
from google import genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()


class GeminiRAG:

    def __init__(self):
        self.client = genai.Client(
            api_key=os.getenv("GOOGLE_API_KEY")
        )

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.vectorstore = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )

    def preguntar(self, pregunta):

        docs = self.vectorstore.similarity_search(pregunta, k=4)

        contexto = "\n\n".join(
            doc.page_content for doc in docs
        )

        prompt = f"""
Responde SOLO utilizando la informacion entregada.

Contexto:
{contexto}

Pregunta:
{pregunta}
"""

        respuesta = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        return respuesta.text
