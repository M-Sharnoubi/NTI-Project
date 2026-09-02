from langchain_core.tools import tool

# ============================================================
# MOCK ORDERS DATABASE
# ============================================================

MOCK_ORDERS = {
    "1001": {
        "status": "تم الشحن",
        "carrier": "أرامكس",
        "expected_delivery": "غدًا"
    },

    "1002": {
        "status": "قيد التجهيز",
        "carrier": None,
        "expected_delivery": "خلال 3 أيام"
    }
}


# ============================================================
# TOOLS
# ============================================================

@tool
def track_order_status(order_id: str) -> str:
    """
    معرفة حالة الطلب وموعد التوصيل المتوقع.
    """

    order_id = str(order_id).strip()

    order = MOCK_ORDERS.get(order_id)

    if not order:
        return f"لم يتم العثور على الطلب رقم {order_id}."

    response = (
        f"حالة الطلب {order_id}: {order['status']}."
    )

    if order["carrier"]:
        response += (
            f" شركة الشحن: {order['carrier']}."
        )

    if order["expected_delivery"]:
        response += (
            f" موعد التوصيل المتوقع: "
            f"{order['expected_delivery']}."
        )

    return response


@tool
def cancel_order(order_id: str) -> str:
    """
    إلغاء الطلب إذا كان مسموحًا بذلك.
    """

    order_id = str(order_id).strip()

    order = MOCK_ORDERS.get(order_id)

    if not order:
        return f"لم يتم العثور على الطلب رقم {order_id}."

    if order["status"] == "تم الشحن":
        return (
            f"لا يمكن إلغاء الطلب {order_id} "
            "لأنه تم شحنه بالفعل."
        )

    order["status"] = "تم الإلغاء"

    return (
        f"تم إلغاء الطلب {order_id} بنجاح."
    )


tools = [
    track_order_status,
    cancel_order
]
