from typing import Optional, List, Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agent.agent import convert_chat_history
from shared.llm import llm

# ============================================================
# RESOLUTION CHECKER
# ============================================================

resolution_template = """
أنت مسؤول عن تقييم هل مشكلة العميل تم حلها أم لا.

مهمتك هي تحديد هل إجابة المساعد حلت مشكلة العميل أم لا.

أجب بكلمة واحدة فقط:

SOLVED

أو:

UNSOLVED


القواعد:

اعتبر المشكلة SOLVED إذا:
- إجابة المساعد قدمت حلًا واضحًا.
- أو قدمت معلومة كافية للإجابة على سؤال العميل.
- أو نفذت الإجراء المطلوب بنجاح.


اعتبر المشكلة UNSOLVED إذا:
- لم تتم الإجابة على سؤال العميل.
- الإجابة غير واضحة.
- المعلومات غير كافية.
- العميل يحتاج إلى إجراء إضافي ولم يتم توضيحه.
- المساعد لم يتمكن من حل المشكلة.


سياق المحادثة السابقة:
{chat_history}


سؤال العميل الحالي:
{question}


إجابة المساعد:
{answer}


التقييم:
"""

resolution_prompt = ChatPromptTemplate.from_template(
    resolution_template
)


resolution_chain = (
    resolution_prompt
    | llm
    | StrOutputParser()
)


def check_resolution(
    user_query: str,
    answer: str,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Check whether the customer's issue was resolved.
    """

    history_messages = convert_chat_history(
        chat_history
    )

    result = resolution_chain.invoke({
        "chat_history": history_messages,
        "question": user_query,
        "answer": answer
    })

    result = result.strip().upper()

    if "SOLVED" in result:
        return "SOLVED"

    return "UNSOLVED"

