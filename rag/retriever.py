from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from agent.agent import convert_chat_history
from typing import List, Dict, Optional
from rag.vectorstore import vectorstore
from shared.llm import llm

# ============================================================
# RETRIEVER
# ============================================================

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# ============================================================
# HELPER: FORMAT DOCUMENTS
# ============================================================

def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# ============================================================
# RAG PROMPT
# ============================================================

template_string = """
أنت Customer Service AI لشركة الأمل.

استخدم الـCONTEXT للإجابة على سؤال العميل.

القواعد:
1. اعتمد على الـCONTEXT عندما يكون السؤال متعلقًا بسياسات الشركة.
2. لا تخترع معلومات غير موجودة في الـCONTEXT.
3. لو المعلومات غير موجودة، قل للعميل إنك لا تملك هذه المعلومة.
4. استخدم اللغة العربية الواضحة.
5. راعي سياق المحادثة السابقة.
6. لا تكرر معلومات غير ضرورية.

سياق المحادثة السابقة:
{chat_history}

CONTEXT:
{context}

سؤال العميل:
{question}

الإجابة:
"""

rag_prompt = ChatPromptTemplate.from_template(template_string)


# ============================================================
# HISTORY-AWARE QUERY REWRITING
# ============================================================

contextualize_template = """
أنت مساعد مسؤول عن إعادة صياغة أسئلة العملاء.

مهمتك هي تحويل سؤال العميل الحالي إلى سؤال مستقل
وواضح يمكن استخدامه في البحث داخل قاعدة البيانات.

القواعد:
1. إذا كان السؤال يعتمد على المحادثة السابقة، استخدم المعلومات
   الموجودة في المحادثة لإعادة صياغته كسؤال مستقل.
2. إذا كان السؤال واضحًا ومستقلًا بالفعل، أعده كما هو.
3. لا تجب عن السؤال.
4. لا تضف أي معلومات غير موجودة في المحادثة.
5. أخرج السؤال المعاد صياغته فقط، بدون شرح.

مثال:

المحادثة السابقة:
العميل: ما هي سياسة الاسترجاع؟
المساعد: يمكن إرجاع المنتج خلال 14 يومًا.

السؤال الحالي:
هل ينفع أرجعه بعد أسبوع؟

السؤال المستقل:
هل يمكن للعميل إرجاع المنتج بعد 7 أيام؟

--------------------------------

المحادثة السابقة:
{chat_history}

السؤال الحالي:
{question}

السؤال المستقل:
"""

contextualize_prompt = ChatPromptTemplate.from_template(
    contextualize_template
)

contextualize_chain = (
    contextualize_prompt
    | llm
    | StrOutputParser()
)


# ============================================================
# RAG CONTEXT FUNCTION
# ============================================================

def get_rag_context(inputs):
    question = inputs["question"]
    chat_history = inputs.get("chat_history", [])


    if chat_history:
        search_query = contextualize_chain.invoke({
            "chat_history": chat_history,
            "question": question
        })
    else:
        search_query = question

    docs = retriever.invoke(search_query)

    return format_docs(docs)


# ============================================================
# RAG CHAIN
# ============================================================

qa_chain = (
    {
        "context": RunnableLambda(get_rag_context),

        "question": lambda x: x["question"],

        "chat_history": lambda x: x.get(
            "chat_history",
            []
        ),
    }

    | rag_prompt
    | llm
    | StrOutputParser()
)

# ============================================================
# RAG PIPELINE
# ============================================================

def run_rag_pipeline(
    user_query: str,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Run the RAG pipeline.

    chat_history is passed by the API layer.
    """

    history_messages = convert_chat_history(
        chat_history
    )

    result = qa_chain.invoke({
        "question": user_query,
        "chat_history": history_messages
    })

    return result
