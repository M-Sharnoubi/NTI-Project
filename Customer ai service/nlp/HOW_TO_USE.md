# How to Use the Trained Model

This guide is for whoever integrates this model into an API, router, or application
(FastAPI, Streamlit, a backend service, etc.). You do **not** need to run training or
`main.py` — you only need the checkpoint folder and the package code.

---

## 1. What you need

You need two things:

1. **The checkpoint folder** — `marbert_multitask_checkpoint/`, containing:
   - `model_state_dict.pt` — the trained model weights
   - `config.json` — label mappings, max sequence length, and model settings
   - tokenizer files (`tokenizer_config.json`, `vocab.txt`, etc.)

2. **The package code** — at minimum:
   - `config.py`
   - `model.py`
   - `preprocessing.py`
   - `checkpoint.py`
   - `inference.py`
   - `__init__.py`

   You do **not** need `train.py`, `metrics.py`, `dataset.py`, or `scripts/main.py` — those
   are only used for training/evaluation.

Install the dependencies:

```bash
pip install torch transformers
```

---

## 2. Load the model once, not per-request

The most important rule: **load the model a single time when your server starts**, and
reuse it for every request. Do not call the inference function without passing
`model`/`tokenizer`/`config` — if you do, it will silently reload the entire model from
disk on every single call, which is very slow.

```python
from customer_service_marbert import load_trained_model

model, tokenizer, config = load_trained_model("marbert_multitask_checkpoint")
```

Do this once at application startup (e.g. in FastAPI's `startup` event, or as a module-level
global loaded when the server process boots).

---

## 3. Run inference on a message

```python
from customer_service_marbert import analyze_customer_message

result = analyze_customer_message(
    "الاوردر #1001 لسه موصلش وأنا متضايق",
    model=model,
    tokenizer=tokenizer,
    config=config,
)

print(result)
```

Output:

```json
{
  "intent": "track_order",
  "sentiment": "negative",
  "priority": "high",
  "entities": {
    "order_id": "#1001"
  },
  "intent_confidence": 0.9421,
  "sentiment_confidence": 0.8873,
  "priority_confidence": 0.7650
}
```

- `intent`, `sentiment`, `priority` — predicted labels.
- `entities.order_id` — extracted order number, or `null` if none was found in the text.
- `*_confidence` fields only appear if `RETURN_CONFIDENCE = True` in `config.py` (enabled by
  default).

---

## 4. Example: FastAPI integration

```python
from fastapi import FastAPI
from pydantic import BaseModel

from customer_service_marbert import load_trained_model, analyze_customer_message

app = FastAPI()

model, tokenizer, config = load_trained_model("marbert_multitask_checkpoint")


class MessageRequest(BaseModel):
    text: str


@app.post("/analyze")
def analyze(request: MessageRequest):
    return analyze_customer_message(
        request.text,
        model=model,
        tokenizer=tokenizer,
        config=config,
    )
```

Run it with:

```bash
uvicorn app:app --reload
```

Then call it:

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "السماعة وصلت مكسورة وأنا محتاج أبدلها"}'
```

---

## 5. Handling low-confidence predictions

If you want to route uncertain predictions to a human agent instead of trusting the model
blindly, check the confidence scores yourself on the caller side:

```python
result = analyze_customer_message(text, model=model, tokenizer=tokenizer, config=config)

LOW_CONFIDENCE_THRESHOLD = 0.5

needs_human_review = any(
    result.get(key, 1.0) < LOW_CONFIDENCE_THRESHOLD
    for key in ("intent_confidence", "sentiment_confidence", "priority_confidence")
)
```

---

## 6. Common errors

| Error | Cause | Fix |
|---|---|---|
| `FileNotFoundError: Checkpoint directory not found` | Wrong path passed to `load_trained_model` | Pass the correct path to `marbert_multitask_checkpoint/`, or use an absolute path |
| `FileNotFoundError: Missing config` / `Missing weights` | Checkpoint folder is incomplete | Make sure `model_state_dict.pt` and `config.json` were both copied |
| `ValueError: Input text is empty after preprocessing` | The input text is empty, whitespace-only, or removed entirely by cleaning | Validate input before calling the function; return an early response for empty text |
| Very slow response time on every request | Model is being reloaded per-request | Load `model`/`tokenizer`/`config` once at startup and reuse them (see Step 2) |
| Model runs on CPU when a GPU is available | `torch.cuda.is_available()` returned `False` in the environment | Check that CUDA drivers and the correct PyTorch build are installed on the server |

---

## 7. Notes

- The model expects **Arabic text**. Text is cleaned automatically via `preprocess_text`
  before tokenization — you don't need to pre-clean it yourself.
- Order-ID extraction recognizes formats like `#1001`, `ORD-1001`, `order 1001`,
  `اوردر رقم 1001`, and `طلب 1001`. If your messages use a different format, extend the
  regex in `inference.py`.
- Max sequence length, label mappings, and other settings used at training time are stored
  in `config.json` inside the checkpoint folder — they are loaded automatically and don't
  need to be set manually.
