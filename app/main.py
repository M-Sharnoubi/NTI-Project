import streamlit as st
import sys
from pathlib import Path

# Add project root directory to Python's import search path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from styles import CUSTOM_CSS
from chat import render_chat
from dashboard import render_dashboard

st.set_page_config(
    page_title="مركز الدعم الفني",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Apply global light design system
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Minimal Header
st.markdown(
    """
    <div class="app-header">
        <div>
            <div class="app-title">متجر الأمل</div>
            <div class="app-subtitle">نظام خدمة العملاء والدعم الفني</div>
        </div>
        <div class="status-pill">
            <div class="status-dot"></div>
            <span>النظام متصل</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_chat, tab_dashboard = st.tabs(["محادثة الدعم", "لوحة التحكم"])

with tab_chat:
    render_chat()

with tab_dashboard:
    render_dashboard()