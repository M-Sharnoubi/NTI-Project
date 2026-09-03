from datetime import datetime
from typing import Dict, List, Any, Optional
from escalation.summarizer import summarize_conversation

# In-memory database storing escalation tickets
tickets_db: List[Dict[str, Any]] = []


def create_support_ticket(
    user_message: str,
    intent: str = "complaint",
    order_id: Optional[str] = None,
    priority: str = "medium",
    conversation_history_text: Optional[str] = None
) -> Dict[str, Any]:
    """
    Creates an escalation ticket for human support and triggers LLM summarization.
    """
    ticket_number = len(tickets_db) + 101
    ticket_id = f"TICK-{ticket_number}"

    # Generate summary using Person 4's summarizer if context exists
    if conversation_history_text:
        try:
            summary_obj = summarize_conversation(conversation_history_text)
            summary_text = summary_obj.summary
        except Exception:
            summary_text = f"طلب غير محلول متصل بـ: {intent}"
    else:
        summary_text = f"بلاغ عن مشكلة مباشرة: {intent}"

    ticket_entry = {
        "ticket_id": ticket_id,
        "user_message": user_message,
        "intent": intent,
        "order_id": order_id,
        "priority": priority,
        "summary": summary_text,
        "status": "OPEN",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    tickets_db.append(ticket_entry)

    return {
        "status": "SUCCESS",
        "message": f"تم فتح تذكرة دعم برقم ({ticket_id}) وتحويلها لموظف خدمة العملاء.",
        "ticket": ticket_entry
    }


def get_all_tickets() -> List[Dict[str, Any]]:
    """Retrieves all open tickets for the Streamlit Admin Dashboard."""
    return tickets_db
