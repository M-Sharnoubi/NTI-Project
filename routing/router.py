"""
Dynamic Router Orchestrator.

This module acts as the central execution hub for the customer support system.
It inspects NLP intent classifications and dynamically routes queries to:
  1. Simple Route: Direct static responses for greetings, thanks, and goodbyes.
  2. RAG Route: Knowledge base retrieval via vector search (rag_agent_module_2).
  3. Agent Route: Tool-assisted actions like order tracking and cancellations.
  4. Escalation Route: Ticket generation for complaints and unresolved queries.
"""

import logging
from typing import Dict, Any, List, Optional

# Import Escalation Logic from escalation package
from escalation.ticket_manager import create_support_ticket

# Import Person 1's NLP module function from inference.py
from nlp.inference import analyze_customer_message

# Import Person 2's RAG & Agent pipelines
from rag.retriever import run_rag_pipeline
from agent.agent import run_agent_pipeline
from routing.resolution_check import check_resolution

# Configure logger for system tracing
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def route_request(
    user_input: str,
    nlp_output: Optional[Dict[str, Any]] = None,
    thread_id: str = "default_session",
    chat_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Orchestrates user requests across simple, RAG, agent, and escalation routes.

    Args:
        user_input (str): Raw customer text input.
        nlp_output (Dict[str, Any], optional): Classified intent and entities. Auto-invokes NLP if None.
        thread_id (str): Unique session identifier for conversation persistence.
        chat_history (List[Dict[str, str]], optional): List of prior chat turns.

    Returns:
        Dict[str, Any]: Unified payload containing the final answer, route used, status, and ticket details.
    """
    # Automatically execute Person 1's NLP module if nlp_output is not provided
    if nlp_output is None:
        try:
            nlp_output = analyze_customer_message(user_input)
        except Exception as err:
            logging.error(f"NLP execution failed in inference.py: {err}")
            nlp_output = {"intent": "general_inquiry", "entities": {}}

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
        # Route 1: Simple Route (Static responses without LLM/RAG overhead)
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

        # Route 3: Agent Route (Executing tool actions like tracking or canceling orders)
        elif intent in ["track_order", "cancel_order"]:
            payload["route_used"] = "AGENT_ROUTE"
            payload["final_answer"] = run_agent_pipeline(user_query=user_input, thread_id=thread_id)

        # Route 4: Escalation Route (Direct human agent escalation)
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

        # Automatic ticket creation if issue remains UNSOLVED
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


# Local execution test block
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Integrated Dynamic Router Module")
    print("=" * 60)

    # Test Case 1: Simple Greeting
    res_1 = route_request("السلام عليكم", nlp_output={"intent": "greeting"})
    print("\n[Test 1 Output - Simple Route]:")
    print(res_1)

    # Test Case 2: Agent Order Tracking
    res_2 = route_request(
        "عايز أعرف شحنة 1001 فين؟",
        nlp_output={"intent": "track_order", "entities": {"order_id": "1001"}},
        thread_id="test_session_1"
    )
    print("\n[Test 2 Output - Agent Route]:")
    print(res_2)
