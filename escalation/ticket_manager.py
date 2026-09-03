"""
Ticket Manager Module (Escalation Engine).

This module handles the creation, storage, and retrieval of customer support 
escalation tickets. It interfaces directly with the LLM summarizer 
(summarizer.py) and automatically calculates ticket priorities based on 
customer sentiment analysis.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional

# Import Person 4's summarizer module from the escalation package
from escalation.summarizer import summarize_conversation

# In-memory database storing escalation tickets for the session
tickets_db: List[Dict[str, Any]] = []


def create_support_ticket(
    user_message: str,
    intent: str = "complaint",
    sentiment: str = "neutral",
    priority: Optional[str] = None,
    order_id: Optional[str] = None,
    conversation_history_text: Optional[str] = None
) -> Dict[str, Any]:
    """
    Creates an escalation ticket for human customer support, calculates priority 
    from sentiment, and triggers LLM summarization.
    """
    ticket_number = len(tickets_db) + 101
    ticket_id = f"TICK-{ticket_number}"

    # Determine priority based on customer sentiment if not explicitly provided
    if not priority:
        sentiment_clean = str(sentiment).lower().strip()
        if sentiment_clean in ["angry", "negative", "frustrated"]:
            priority = "HIGH"
        elif sentiment_clean in ["neutral"]:
            priority = "MEDIUM"
        else:
            priority = "LOW"
    else:
        priority = priority.upper()

    # Generate an automated summary using Person 4's summarizer if context exists
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
        "sentiment": sentiment,
        "priority": priority,
        "order_id": order_id,
        "summary": summary_text,
        "status": "OPEN",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    tickets_db.append(ticket_entry)

    return {
        "status": "SUCCESS",
        "message": f"تم فتح تذكرة دعم برقم ({ticket_id}) بأولوية ({priority}) وتحويلها لموظف خدمة العملاء.",
        "ticket": ticket_entry
    }


def get_all_tickets() -> List[Dict[str, Any]]:
    """
    Retrieves all open tickets for Person 4's Streamlit Admin Dashboard.
    """
    return tickets_db
