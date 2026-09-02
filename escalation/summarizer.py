import os
from dotenv import load_dotenv
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_classic.docstore.document import Document
from langchain_classic.prompts import PromptTemplate
from langchain_groq import ChatGroq

from schemas.contracts import SummaryResult

load_dotenv()

SUMMARY_PROMPT = PromptTemplate(
    template=(
        "You are summarizing an unresolved Arabic customer support conversation for a human agent.\n"
        "Write a short, factual summary (2-3 sentences max) in Arabic covering:\n"
        "- What the customer wants\n"
        "- What was tried or blocked\n"
        "- Any relevant order/product IDs\n\n"
        "Conversation:\n{text}\n\n"
        "Summary:"
    ),
    input_variables=["text"],
)


def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        api_key=os.environ.get("GROQ_API_KEY"),
    )


def summarize_conversation(conversation_text: str) -> SummaryResult:
    """
    conversation_text: full text of the conversation (customer + bot turns) that ended unresolved.
    Returns a SummaryResult matching the shared contract.
    """
    llm = get_llm()
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=SUMMARY_PROMPT)

    doc = Document(page_content=conversation_text)
    result = chain.invoke({"input_documents": [doc]})

    summary_text = result["output_text"].strip()
    return SummaryResult(summary=summary_text)


if __name__ == "__main__":
    # quick manual test
    sample = (
        "العميل: عايز ألغي طلب رقم 12345\n"
        "البوت: للأسف مش قادر أتحقق من أهلية الاسترجاع للطلب ده.\n"
        "العميل: طب أعمل ايه؟\n"
        "البوت: هحولك لموظف خدمة عملاء."
    )
    print(summarize_conversation(sample))
