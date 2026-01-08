import os
import logging
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer
from opencensus.ext.azure.log_exporter import AzureLogHandler

# =========================
# CONFIG LOGGING (AZURE)
# =========================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

instrumentation_key = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
if instrumentation_key:
    logger.addHandler(
        AzureLogHandler(connection_string=instrumentation_key)
    )

# =========================
# FASTAPI APP
# =========================
app = FastAPI(
    title="NLP Emotion Detection API",
    version="1.0.0"
)

# =========================
# LABELS
# =========================
EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
VIOLENCE_LABELS = ["non_violent", "violent"]
HATE_LABELS = ["non_hate", "hate"]

# =========================
# SCHEMAS
# =========================
class NLPRequest(BaseModel):
    text: str


class TaskResult(BaseModel):
    label: str
    confidence: float


class NLPResponse(BaseModel):
    emotion: TaskResult
    violence: TaskResult
    hate: TaskResult


# =========================
# LOAD MODEL
# =========================
MODEL_PATH = "models/final_model_projet4/model.pt"
TOKENIZER_NAME = "distilbert-base-uncased"

model = None
tokenizer = None


@app.on_event("startup")
def load_model():
    global model, tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        model = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
        model.eval()
        logger.info("ModelLoadedSuccessfully")
    except Exception as e:
        logger.error(f"ModelLoadingFailed: {str(e)}")


# =========================
# ROUTES
# =========================
@app.get("/")
def root():
    return {"status": "API is running"}


@app.post("/predict", response_model=NLPResponse)
def predict(request: NLPRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

    result = {}

    for task, labels in [
        ("emotion", EMOTION_LABELS),
        ("violence", VIOLENCE_LABELS),
        ("hate", HATE_LABELS),
    ]:
        probs = torch.softmax(outputs[task], dim=-1)
        confidence, index = torch.max(probs, dim=-1)

        result[task] = TaskResult(
            label=labels[index.item()],
            confidence=float(confidence.item())
        )

    # =========================
    # AZURE LOG
    # =========================
    logger.info(
        "PredictionMade",
        extra={
            "custom_dimensions": {
                "text_preview": request.text[:30],
                "emotion": result["emotion"].label,
                "confidence": result["emotion"].confidence
            }
        }
    )

    return NLPResponse(**result)
