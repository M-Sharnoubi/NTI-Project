import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver


# ============================================================
# 1. ENVIRONMENT
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is not set. "
        "Please add it to your .env file."
    )


# ============================================================
# 2. LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    groq_api_key=GROQ_API_KEY,
)


# ============================================================
# 3. EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# ============================================================
# 4. LOAD COMPANY POLICY
# ============================================================

loader = TextLoader(
    "company_policies.txt",
    encoding="utf-8"
)

documents = loader.load()


# ============================================================
# 5. SPLIT DOCUMENTS
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)


# ============================================================
# 6. CHROMA VECTOR DATABASE
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


# ============================================================
# 7. RETRIEVER
# ============================================================

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# ============================================================
# 8. HELPER: FORMAT DOCUMENTS
# ============================================================

def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# ============================================================
# 9. RAG PROMPT
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
# 10. HISTORY-AWARE QUERY REWRITING
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
# 11. RAG CONTEXT FUNCTION
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
# 12. RAG CHAIN
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
# 13. MOCK ORDERS DATABASE
# ============================================================

MOCK_ORDERS = {
    "1001": {
        "status": "تم الشحن",
        "carrier": "أرامكس",
        "expected_delivery": "غدًا"
    },

    "1002": {
        "status": "قيد التجهيز",
        "carrier": None,
        "expected_delivery": "خلال 3 أيام"
    }
}


# ============================================================
# 14. TOOLS
# ============================================================

@tool
def track_order_status(order_id: str) -> str:
    """
    معرفة حالة الطلب وموعد التوصيل المتوقع.
    """

    order_id = str(order_id).strip()

    order = MOCK_ORDERS.get(order_id)

    if not order:
        return f"لم يتم العثور على الطلب رقم {order_id}."

    response = (
        f"حالة الطلب {order_id}: {order['status']}."
    )

    if order["carrier"]:
        response += (
            f" شركة الشحن: {order['carrier']}."
        )

    if order["expected_delivery"]:
        response += (
            f" موعد التوصيل المتوقع: "
            f"{order['expected_delivery']}."
        )

    return response


@tool
def cancel_order(order_id: str) -> str:
    """
    إلغاء الطلب إذا كان مسموحًا بذلك.
    """

    order_id = str(order_id).strip()

    order = MOCK_ORDERS.get(order_id)

    if not order:
        return f"لم يتم العثور على الطلب رقم {order_id}."

    if order["status"] == "تم الشحن":
        return (
            f"لا يمكن إلغاء الطلب {order_id} "
            "لأنه تم شحنه بالفعل."
        )

    order["status"] = "تم الإلغاء"

    return (
        f"تم إلغاء الطلب {order_id} بنجاح."
    )


tools = [
    track_order_status,
    cancel_order
]


# ============================================================
# 15. AGENT MEMORY
# ============================================================

# InMemorySaver gives the agent short-term memory.
#
# Important:
# This memory exists while the Python process is running.
# For production, replace it with a persistent checkpointer.

checkpointer = InMemorySaver()


# ============================================================
# 16. AGENT
# ============================================================

agent_executor = create_agent(
    model=llm,
    tools=tools,

    system_prompt="""
أنت Customer Service AI لشركة الأمل.

وظيفتك:
- مساعدة العملاء.
- الإجابة عن أسئلتهم.
- تتبع الطلبات.
- إلغاء الطلبات عندما يكون ذلك مسموحًا.
- استخدام سياق المحادثة السابقة.

القواعد:
1. استخدم الأدوات عندما تحتاج لمعلومات عن الطلبات.
2. لا تخترع حالة طلب.
3. إذا أعطى العميل رقم طلب في رسالة سابقة،
   يمكنك استخدامه عند سؤاله سؤالًا تابعًا.
4. حافظ على سياق المحادثة.
5. كن واضحًا ومختصرًا.
6. تحدث بالعربية.
""",

    checkpointer=checkpointer
)


# ============================================================
# 17. CONVERT HISTORY
# ============================================================

def convert_chat_history(
    chat_history: Optional[List[Dict[str, str]]]
):
    """
    Convert API-style history dictionaries
    into LangChain messages.
    """

    if not chat_history:
        return []

    messages = []

    for message in chat_history:

        role = message.get("role")
        content = message.get("content", "")

        if role == "user":
            messages.append(
                HumanMessage(content=content)
            )

        elif role == "assistant":
            messages.append(
                AIMessage(content=content)
            )

    return messages


# ============================================================
# 18. RAG PIPELINE
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


# ============================================================
# 19. AGENT PIPELINE
# ============================================================

def run_agent_pipeline(
    user_query: str,
    thread_id: str
) -> str:
    """
    Run the agent using persistent conversation
    state identified by thread_id.
    """

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = agent_executor.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_query
                }
            ]
        },
        config
    )

    final_message = result["messages"][-1]

    if hasattr(final_message, "content"):
        return final_message.content

    return str(final_message)


# ============================================================
# 20. RESOLUTION CHECKER
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


# ============================================================
# 21. TEST RAG
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TEST 1: RAG")
    print("=" * 60)

    rag_answer = run_rag_pipeline(
        "ما هي سياسة الاسترجاع؟"
    )

    print("\nRAG Answer:")
    print(rag_answer)


    # ========================================================
    # TEST AGENT MEMORY
    # ========================================================

    print("\n" + "=" * 60)
    print("TEST 2: AGENT MEMORY")
    print("=" * 60)

    thread_id = "customer_001"

    # First message
    answer_1 = run_agent_pipeline(
        "أنا عايز استعلم عن الطلب رقم 1001",
        thread_id=thread_id
    )

    print("\nFirst Answer:")
    print(answer_1)


    # Follow-up message
    answer_2 = run_agent_pipeline(
        "هو هتوصل لي أمتى؟",
        thread_id=thread_id
    )

    print("\nFollow-up Answer:")
    print(answer_2)


    # ========================================================
    # TEST RESOLUTION
    # ========================================================

    print("\n" + "=" * 60)
    print("TEST 3: RESOLUTION")
    print("=" * 60)

    resolution = check_resolution(
        user_query="هو هتوصل لي أمتى؟",
        answer=answer_2
    )

    print("\nResolution:")
    print(resolution)
