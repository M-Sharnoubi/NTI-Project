import streamlit as st

from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
FILE_PATH = BASE_DIR / "company_policies.txt"
CHROMA_DIR = BASE_DIR / "chroma_db"


# ============================================================
# LOAD COMPANY POLICY
# ============================================================

loader = TextLoader(
    str(FILE_PATH),
    encoding="utf-8"
)

documents = loader.load()


# ============================================================
# SPLIT DOCUMENTS
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)


# ============================================================
# EMBEDDINGS + VECTORSTORE
# ============================================================

@st.cache_resource(
    show_spinner="جاري تحميل قاعدة المعرفة..."
)
def get_vectorstore():

    # --------------------------------------------------------
    # Load embedding model only once
    # --------------------------------------------------------

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # --------------------------------------------------------
    # Create / load Chroma vector database
    # --------------------------------------------------------

    vectorstore = Chroma(
        collection_name="al_amal_policy",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    # --------------------------------------------------------
    # Add company policies only if collection is empty
    # --------------------------------------------------------

    collection_data = vectorstore.get()

    if not collection_data["ids"]:
        vectorstore.add_documents(chunks)

    return vectorstore


# ============================================================
# SHARED VECTORSTORE
# ============================================================

vectorstore = get_vectorstore()
