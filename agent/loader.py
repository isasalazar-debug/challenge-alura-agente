from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # ✅ CORREGIDO
import os


def cargar_documentos(ruta_pdf):
    """
    Carga un PDF, lo divide en chunks y retorna los documentos listos para FAISS.
    """
    if not os.path.exists(ruta_pdf):
        raise FileNotFoundError(
            f"No se encontró el PDF: {ruta_pdf}\n"
            f"Asegúrate de que esté en la carpeta 'data/'."
        )

    loader = PyPDFLoader(ruta_pdf)
    paginas = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    documentos = splitter.split_documents(paginas)

    print(f"📄 PDF cargado: {ruta_pdf}")
    print(f"📑 Páginas originales: {len(paginas)}")
    print(f"🧩 Chunks generados: {len(documentos)}")

    return documentos