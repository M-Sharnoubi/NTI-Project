"""
Admin Dashboard — Open tickets, priority management, filtering, and status updates.
"""

import pandas as pd
import streamlit as st
from escalation.ticket_manager import get_all_tickets

# Priority and Sentiment Arabic Mapping
PRIORITY_LABELS_AR = {"high": "عالية", "medium": "متوسطة", "low": "منخفضة"}
SENTIMENT_LABELS_AR = {
    "angry": "غاضب",
    "frustrated": "محبط",
    "neutral": "محايد",
    "happy": "سعيد",
}


def render_dashboard():
    # 1. Header & Quick Controls
    col_head, col_act1, col_act2 = st.columns([3, 1, 1])
    with col_head:
        st.title("لوحة تحكم الدعم الفني")
    with col_act1:
        if st.button("تحديث البيانات", use_container_width=True):
            st.rerun()

    # Load Tickets
    with st.spinner("جاري تحميل التذاكر..."):
        tickets = get_all_tickets()

    if not tickets:
        st.info("لا توجد تذاكر مفتوحة حالياً.")
        return

    # Add CSV Export in Action Bar
    with col_act2:
        df_export = pd.DataFrame(tickets)
        st.download_button(
            label="تصدير (CSV)",
            data=df_export.to_csv(index=False).encode("utf-8-sig"),
            file_name="support_tickets.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # 2. Metrics Summary Bar
    high_count = sum(1 for t in tickets if str(t.get("priority")).lower() == "high")
    med_count = sum(1 for t in tickets if str(t.get("priority")).lower() == "medium")
    total_count = len(tickets)

    st.markdown(
        f"""
        <div class="metrics-grid" style="display: flex; gap: 15px; margin-bottom: 20px;">
            <div class="metric-card" style="flex: 1; padding: 15px; border-radius: 8px; background: rgba(255,255,255,0.05); text-align: center;">
                <div class="metric-value" style="font-size: 1.8rem; font-weight: bold;">{total_count}</div>
                <div class="metric-label" style="color: gray;">إجمالي التذاكر</div>
            </div>
            <div class="metric-card" style="flex: 1; padding: 15px; border-radius: 8px; background: rgba(255,255,255,0.05); text-align: center;">
                <div class="metric-value" style="font-size: 1.8rem; font-weight: bold; color: var(--danger, #ff4d4f);">{high_count}</div>
                <div class="metric-label" style="color: gray;">أولوية عالية</div>
            </div>
            <div class="metric-card" style="flex: 1; padding: 15px; border-radius: 8px; background: rgba(255,255,255,0.05); text-align: center;">
                <div class="metric-value" style="font-size: 1.8rem; font-weight: bold; color: var(--warning, #faad14);">{med_count}</div>
                <div class="metric-label" style="color: gray;">أولوية متوسطة</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Search & Interactive Filter Bar
    col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
    with col_f1:
        search_query = st.text_input("بحث برقم التذكرة أو رقم الطلب:", "")
    with col_f2:
        priority_filter = st.multiselect(
            "تصفية حسب الأولوية:",
            options=["high", "medium", "low"],
            format_func=lambda x: PRIORITY_LABELS_AR.get(x, x),
        )
    with col_f3:
        status_filter = st.selectbox(
            "تصفية حسب الحالة:",
            ["الكل", "OPEN", "IN_PROGRESS", "RESOLVED", "CLOSED"],
        )

    # Sorting Logic (High Priority First)
    priority_order = {"high": 0, "medium": 1, "low": 2}
    tickets_sorted = sorted(
        tickets, key=lambda t: priority_order.get(str(t.get("priority")).lower(), 3)
    )

    # Apply Filters
    filtered_tickets = tickets_sorted
    if search_query.strip():
        q = search_query.strip().lower()
        filtered_tickets = [
            t for t in filtered_tickets
            if q in str(t.get("ticket_id", "")).lower() or q in str(t.get("order_id", "")).lower()
        ]
    if priority_filter:
        filtered_tickets = [
            t for t in filtered_tickets
            if str(t.get("priority")).lower() in priority_filter
        ]
    if status_filter != "الكل":
        filtered_tickets = [
            t for t in filtered_tickets
            if str(t.get("status")).upper() == status_filter
        ]

    st.markdown("---")

    if not filtered_tickets:
        st.warning("لا توجد تذاكر تطابق معايير البحث والتصفية.")
        return

    # 4. Ticket Cards List
    for t in filtered_tickets:
        ticket_id = t.get("ticket_id", "—")
        raw_priority = str(t.get("priority", "low")).lower()
        priority_ar = PRIORITY_LABELS_AR.get(raw_priority, raw_priority)
        status = t.get("status", "OPEN")
        raw_sentiment = str(t.get("sentiment", "neutral")).lower()
        sentiment_ar = SENTIMENT_LABELS_AR.get(raw_sentiment, raw_sentiment)

        # Ticket Card UI Header
        st.markdown(
            f"""
            <div class="ticket-card" style="padding: 12px; border-radius: 6px; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.1); margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 600; font-size: 1rem;">
                        تذكرة #{ticket_id}
                    </span>
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <span class="priority-badge {raw_priority}" style="padding: 2px 8px; border-radius: 4px; font-size: 0.85rem;">
                            الأولوية: <b>{priority_ar}</b>
                        </span>
                        <span style="font-size: 0.85rem;">
                            الحالة: <b>{status}</b>
                        </span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Ticket Details & Interactive Management
        with st.expander(f"عرض التفاصيل وتحديث الحالة (تذكرة #{ticket_id})"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**ملخص المشكلة:**", t.get("summary", "—"))
                st.write("**رسالة العميل:**", t.get("user_message", "—"))
                st.write("**نية العميل (Intent):**", f"`{t.get('intent', '—')}`")

            with col2:
                if t.get("order_id"):
                    st.write("**رقم الطلب:**", f"`{t['order_id']}`")
                st.write("**درجة الأولوية:**", priority_ar)
                st.write("**شعور العميل (Sentiment):**", sentiment_ar)

            # Conversation History View
            if t.get("conversation_history_text") or t.get("conversation_history"):
                hist = t.get("conversation_history_text") or t.get("conversation_history")
                st.write("**سجل المحادثة:**")
                st.text_area(
                    label="سجل المحادثة",
                    value=hist,
                    height=100,
                    disabled=True,
                    key=f"hist_{ticket_id}",
                    label_visibility="collapsed"
                )

            # Action Bar to Update Status
            st.divider()
            c_select, c_btn = st.columns([3, 1])

            status_options = ["OPEN", "IN_PROGRESS", "RESOLVED", "CLOSED"]
            current_index = status_options.index(status) if status in status_options else 0

            with c_select:
                selected_status = st.selectbox(
                    "تحديث حالة التذكرة:",
                    options=status_options,
                    index=current_index,
                    key=f"status_select_{ticket_id}"
                )

            with c_btn:
                st.write(" ")  # Spacer for visual alignment
                if st.button("حفظ التحديث", key=f"btn_save_{ticket_id}", use_container_width=True):
                    # update_ticket_status(ticket_id, selected_status)
                    st.success(f"تم تغيير حالة التذكرة إلى {selected_status}")
                    st.rerun()


if __name__ == "__main__":
    render_dashboard()