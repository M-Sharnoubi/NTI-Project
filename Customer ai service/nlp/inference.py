"""
Regex entity extraction and the reusable single-message
inference function (usable from FastAPI, a router, Streamlit, etc.).
"""


import re
from contextlib import nullcontext
from typing import Dict, Optional
 
import torch
import torch.nn as nn
 
from .config import DEVICE, USE_AMP, RETURN_CONFIDENCE, CHECKPOINT_DIR
from .preprocessing import preprocess_text
from .checkpoint import load_trained_model
 
 
# ============================================================
# 22. REGEX ENTITY EXTRACTION
# ============================================================
 
# Matches order-id mentions in several common customer-message
# formats, e.g.:
#   #1001
#   ORD-1001 / ORD1001 / order 1001
#   اوردر 1001 / أوردر رقم 1001 / طلب رقم #1001
ORDER_ID_PATTERN = re.compile(
    r"""
    (?<!\w)\#(?P<hash>\d{3,})                       # #1001
    |
    \b(?:ORD|order)[\s\-_]?(?P<latin>\d{3,})\b       # ORD-1001, order 1001
    |
    (?:اوردر|أوردر|طلب)\s*(?:رقم\s*)?\#?(?P<arabic>\d{3,})  # اوردر 1001, طلب رقم 1001
    """,
    re.IGNORECASE | re.VERBOSE,
)
 
 
def extract_entities(
    text: str,
) -> Dict[str, Optional[str]]:
 
    match = (
        ORDER_ID_PATTERN.search(
            text or ""
        )
    )
 
    order_id = None
 
    if match:
 
        # Whichever named group actually matched.
        digits = (
            match.group("hash")
            or match.group("latin")
            or match.group("arabic")
        )
 
        # Normalize to a single consistent display format
        # regardless of how the customer wrote it.
        order_id = f"#{digits}"
 
    return {
        "order_id": order_id
    }
 
# ============================================================
# 23. INFERENCE
# ============================================================

def analyze_customer_message(
    text: str,
    model: Optional[nn.Module] = None,
    tokenizer=None,
    config: Optional[Dict] = None,
) -> Dict:

    """
    Main reusable inference function.

    Can later be called by:
      - FastAPI
      - Router
      - Streamlit
      - another Python application
    """

    # Lazy-load model if caller didn't provide it.
    if (
        model is None
        or tokenizer is None
        or config is None
    ):

        (
            model,
            tokenizer,
            config,
        ) = load_trained_model(
            CHECKPOINT_DIR
        )

    cleaned_text = preprocess_text(
        text
    )

    if not cleaned_text:
        raise ValueError(
            "Input text is empty after preprocessing."
        )

    max_length = int(
        config[
            "max_length"
        ]
    )

    encoded = tokenizer(
        cleaned_text,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )

    encoded = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }

    model.eval()

    with torch.no_grad():

        with (
            torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            )
            if USE_AMP
            else nullcontext()
        ):

            outputs = model(
                input_ids=encoded[
                    "input_ids"
                ],

                attention_mask=encoded[
                    "attention_mask"
                ],
            )

    # Softmax ONLY at inference time.
    intent_probs = torch.softmax(
        outputs[
            "intent_logits"
        ],
        dim=-1,
    )

    sentiment_probs = torch.softmax(
        outputs[
            "sentiment_logits"
        ],
        dim=-1,
    )

    priority_probs = torch.softmax(
        outputs[
            "priority_logits"
        ],
        dim=-1,
    )

    intent_id = int(
        intent_probs
        .argmax(dim=-1)
        .item()
    )

    sentiment_id = int(
        sentiment_probs
        .argmax(dim=-1)
        .item()
    )

    priority_id = int(
        priority_probs
        .argmax(dim=-1)
        .item()
    )

    result = {
        "intent":
            config[
                "id2intent"
            ][str(intent_id)],

        "sentiment":
            config[
                "id2sentiment"
            ][str(sentiment_id)],

        "priority":
            config[
                "id2priority"
            ][str(priority_id)],

        "entities":
            extract_entities(
                cleaned_text
            ),
    }

    if RETURN_CONFIDENCE:

        result.update(
            {
                "intent_confidence":
                    round(
                        float(
                            intent_probs[
                                0,
                                intent_id,
                            ].item()
                        ),
                        4,
                    ),

                "sentiment_confidence":
                    round(
                        float(
                            sentiment_probs[
                                0,
                                sentiment_id,
                            ].item()
                        ),
                        4,
                    ),

                "priority_confidence":
                    round(
                        float(
                            priority_probs[
                                0,
                                priority_id,
                            ].item()
                        ),
                        4,
                    ),
            }
        )

    return result
