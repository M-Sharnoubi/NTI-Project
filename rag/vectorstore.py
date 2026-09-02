from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# ============================================================
# EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# ============================================================
# LOAD COMPANY POLICY
# ============================================================

loader = TextLoader(
    "company_policies.txt",
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
# CHROMA VECTOR DATABASE
# ============================================================

vectorstore = Chroma(
    collection_name="al_amal_policy",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Add documents only if collection is empty
collection_data = vectorstore.get()

if not collection_data["ids"]:
    vectorstore.add_documents(chunks)
