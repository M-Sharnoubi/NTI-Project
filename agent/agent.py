from typing import Optional, List, Dict

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent.tools import tools
from shared.llm import llm

# ============================================================
# AGENT MEMORY
# ============================================================

# InMemorySaver gives the agent short-term memory.
#
# Important:
# This memory exists while the Python process is running.
# For production, replace it with a persistent checkpointer.

checkpointer = InMemorySaver()


# ============================================================
# AGENT
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
# CONVERT HISTORY
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
# AGENT PIPELINE
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

