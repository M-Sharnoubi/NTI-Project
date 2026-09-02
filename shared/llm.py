from langchain_groq import ChatGroq
from shared.config import GROQ_API_KEY

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    groq_api_key=GROQ_API_KEY,
)