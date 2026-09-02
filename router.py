import logging
from typing import Dict, Any, List, Optional

# Import Escalation Logic from escalation package
from escalation.ticket_manager import create_support_ticket

# Import Pipelines from Person 2's module
from rag_agent_module_2 import (
    run_rag_pipeline,
    run_agent_pipeline,
    check_resolution
)

# Configure system logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def route_request(
    user_input: str,
    nlp_output: Optional[Dict[str, Any]] = None,
    thread_id: str = "default_session",
    chat_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Main Dynamic Router orchestrating intent execution across simple, RAG, agent, and escalation flows.
    """
    if nlp_output is None:
        nlp_output = {}

    intent = nlp_output.get("intent", "general_inquiry")
    entities = nlp_output.get("entities", {})
    order_id = entities.get("order_id") if isinstance(entities, dict) else getattr(entities, "order_id", None)

    payload: Dict[str, Any] = {
        "route_used": "",
        "final_answer": "",
        "status": "SOLVED",
        "ticket_info": None
    }

    try:
        # Route 1: Simple Route (Greetings, Thanks, Goodbyes)
        if intent in ["greeting", "thanks", "goodbye"]:
            payload["route_used"] = "SIMPLE_ROUTE"
            payload["status"] = "SOLVED"

            if intent == "greeting":
                payload["final_answer"] = "أهلاً بك في متجر الأمل للإلكترونيات! كيف يمكنني مساعدتك اليوم؟"
            elif intent == "thanks":
                payload["final_answer"] = "العفو! نحن في خدمتك دائماً."
            else:
                payload["final_answer"] = "شكراً لتواصلك مع متجر الأمل للإلكترونيات. نتمنى لك يوماً سعيداً!"

            return payload

        # Route 2: RAG Route (Knowledge Base Lookup)
        elif intent in ["policy_inquiry", "general_inquiry"]:
            payload["route_used"] = "RAG_ROUTE"
            payload["final_answer"] = run_rag_pipeline(user_query=user_input, chat_history=chat_history)

        # Route 3: Agent Route (Order Actions via Tools)
        elif intent in ["track_order", "cancel_order"]:
            payload["route_used"] = "AGENT_ROUTE"
            payload["final_answer"] = run_agent_pipeline(user_query=user_input, thread_id=thread_id)

        # Route 4: Escalation Route (Direct Complaints)
        elif intent == "complaint":
            payload["route_used"] = "ESCALATION_ROUTE"
            payload["status"] = "UNSOLVED"

            ticket_res = create_support_ticket(
                user_message=user_input,
                intent=intent,
                order_id=order_id
            )
            payload["ticket_info"] = ticket_res.get("ticket")
            payload["final_answer"] = (
                f"نأسف للإزعاج. {ticket_res.get('message')} "
                "سيقوم أحد ممثلي خدمة العملاء بالتواصل معك في أقرب وقت."
            )
            return payload

        # Fallback Route
        else:
            payload["route_used"] = "RAG_ROUTE"
            payload["final_answer"] = run_rag_pipeline(user_query=user_input, chat_history=chat_history)

        # Step 5: Resolution Verification
        resolution_status = check_resolution(
            user_query=user_input,
            answer=payload["final_answer"],
            chat_history=chat_history
        )
        payload["status"] = resolution_status

        # If unresolved, generate summary and support ticket
        if resolution_status == "UNSOLVED":
            history_str = f"العميل: {user_input}\nالبوت: {payload['final_answer']}"
            ticket_res = create_support_ticket(
                user_message=user_input,
                intent=intent,
                order_id=order_id,
                conversation_history_text=history_str
            )
            payload["ticket_info"] = ticket_res.get("ticket")
            payload["final_answer"] += f"\n\n(ملاحظة: {ticket_res.get('message')})"

    except Exception as error:
        logging.error(f"Execution Error in Dynamic Router: {str(error)}", exc_info=True)
        payload["route_used"] = "ERROR_FALLBACK"
        payload["status"] = "UNSOLVED"

        ticket_res = create_support_ticket(
            user_message=user_input,
            intent="system_error",
            order_id=order_id
        )
        payload["ticket_info"] = ticket_res.get("ticket")
        payload["final_answer"] = f"عذراً، حدث خطأ غير متوقع. {ticket_res.get('message')}"

    return payload


# Local Testing Block
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Integrated Dynamic Router")
    print("=" * 60)

    # Test Case 1: Simple Greeting
    res_1 = route_request("السلام عليكم", nlp_output={"intent": "greeting"})
    print("\nTest 1 (Greeting):", res_1)

    # Test Case 2: Agent Action
    res_2 = route_request("عايز أعرف شحنة 1001 فين؟", nlp_output={"intent": "track_order", "entities": {"order_id": "1001"}})
    print("\nTest 2 (Track Order):", res_2)
