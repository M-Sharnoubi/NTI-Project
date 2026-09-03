"""
Day 2 update: swapped the dummy pipeline for the real router, added a
persistent thread_id for agent memory, and now forwards chat_history so
RAG's query rewriting and resolution checks have real context.
"""
import uuid
import streamlit as st

from routing.router import route_request


def run_pipeline(user_text: str, thread_id: str, chat_history: list[dict]) -> dict:
    """Adapts route_request()'s output shape to what this UI expects."""
    result = route_request(
        user_input=user_text,
        thread_id=thread_id,
        chat_history=chat_history,
    )
    return {
        "route": result["route_used"],
        "resolved": result["status"] == "SOLVED",
        "response": result["final_answer"],
    }


def render_chat():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    # Professional empty state
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="welcome-container">
                <div class="welcome-title">مرحباً بك في مركز الخدمة</div>
                <div class="welcome-desc">يمكنك الاستفسار عن حالة الطلب، طلبات الاستبدال، أو الدعم الفني المباشر.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Render Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("escalated"):
                st.markdown(
                    """
                    <div class="escalation-box">
                        <span>تم تسجيل تذكرة وسيتم التواصل معك بواسطة أحد ممثلي خدمة العملاء.</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    user_input = st.chat_input("اكتب استفسارك هنا...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("جاري التواصل مع النظام..."):
            result = run_pipeline(
                user_input,
                thread_id=st.session_state.thread_id,
                chat_history=st.session_state.messages[:-1],
            )

        with st.chat_message("assistant"):
            st.write(result["response"])
            is_escalated = not result["resolved"]
            if is_escalated:
                st.markdown(
                    """
                    <div class="escalation-box">
                        <span>تم تسجيل تذكرة وسيتم التواصل معك بواسطة أحد ممثلي خدمة العملاء.</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["response"],
                "escalated": is_escalated,
            }
        )


if __name__ == "__main__":
    render_chat()