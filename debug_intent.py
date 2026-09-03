from nlp.inference import analyze_customer_message

test_messages = [
    "عايز أعرف حالة الطلب رقم 1001",
    "هو هيوصل لي أمتى؟",
    "عايز ألغي الطلب رقم 1001",
    "عايز ألغي الطلب رقم 1002",
]

for msg in test_messages:
    result = analyze_customer_message(msg)
    print(repr(msg))
    print("  ->", result)
    print()