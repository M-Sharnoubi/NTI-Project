"""
Admin Dashboard — open tickets, priority, and summaries.
"""

import streamlit as st
from escalation.ticket_manager import get_all_tickets

PRIORITY_LABELS_AR = {"high": "عالية", "medium": "متوسطة", "low": "منخفضة"}


def render_dashboard():
    with st.spinner("جاري تحميل التذاكر..."):
        tickets = get_all_tickets()

    if not tickets:
        st.info("✨ لا توجد تذاكر مفتوحة حاليًا. جميع الطلبات معالجة!")
        return

    # Metrics Summary Bar
    high_count = sum(1 for t in tickets if t.get("priority") == "high")
    med_count = sum(1 for t in tickets if t.get("priority") == "medium")
    total_count = len(tickets)

    st.markdown(
        f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{total_count}</div>
                <div class="metric-label">إجمالي التذاكر</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="color: var(--danger);">{high_count}</div>
                <div class="metric-label">أولوية عالية</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="color: var(--warning);">{med_count}</div>
                <div class="metric-label">أولوية متوسطة</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    priority_order = {"high": 0, "medium": 1, "low": 2}
    tickets_sorted = sorted(
        tickets, key=lambda t: priority_order.get(t.get("priority", "low"), 3)
    )

    for t in tickets_sorted:
        priority = t.get("priority", "low")
        priority_ar = PRIORITY_LABELS_AR.get(priority, priority)

        st.markdown(
            f"""
            <div class="ticket-card {priority}">
                <div>
                    <span style="font-weight: 800; font-size: 1.1rem; color: var(--text-heading);">
                        🎫 تذكرة #{t.get('ticket_id', '—')}
                    </span>
                </div>
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span class="priority-badge {priority}">{priority_ar}</span>
                    <span style="color: var(--text-muted); font-size: 0.9rem;">
                        الحالة: <b>{t.get('status', '—')}</b>
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("🔎 عرض تفاصيل التذكرة"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**ملخص المشكلة:**", t.get("summary", "—"))
                st.write("**نية العميل:**", t.get("intent", "—"))
            with col2:
                if t.get("order_id"):
                    st.write("**رقم الطلب:**", f"`{t['order_id']}`")
                st.write("**درجة الأولوية:**", priority_ar)