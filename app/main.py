import streamlit as st

from styles import CUSTOM_CSS
from chat import render_chat
from dashboard import render_dashboard

st.set_page_config(
    page_title="متجر الأمل — خدمة العملاء",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Apply unified design system
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Custom App Bar
st.markdown(
    """
    <div class="app-header">
        <div>
            <div class="app-title">متجر الأمل 🛍️</div>
            <div class="app-subtitle">نظام خدمة العملاء الذكي والدعم الفني</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_chat, tab_dashboard = st.tabs(["💬 شات العميل", "📊 لوحة التحكم"])

with tab_chat:
    render_chat()

with tab_dashboard:
    render_dashboard()