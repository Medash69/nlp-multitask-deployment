import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from transformers import AutoModel, AutoTokenizer
from app.models import NLPRequest, NLPResponse, HealthResponse, TaskResult
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. ARCHITECTURE EXACTE (Déduite des erreurs de ton dictionnaire de poids)
class MultiTaskClassifier(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-base"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = 768  # Taille DeBERTa
        head_inter_size = 384 # Taille détectée pour les têtes

        # La shared_layer fait du 768 -> 768 (détecté par l'erreur Size([768, 768]))
        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),      # .0
            nn.LayerNorm(hidden_size),                 # .1
            nn.ReLU(),                                 # .2
            nn.Dropout(0.1),                           # .3
            nn.Linear(hidden_size, hidden_size),      # .4
            nn.LayerNorm(hidden_size)                  # .5
        )

        # Les têtes font du 768 -> 384 -> labels
        def task_head(out_labels):
            return nn.Sequential(
                nn.Linear(hidden_size, head_inter_size), # .0
                nn.LayerNorm(head_inter_size),            # .1
                nn.ReLU(),                                # .2
                nn.Dropout(0.1),                          # .3
                nn.Linear(head_inter_size, out_labels)    # .4
            )

        self.emotion_head = task_head(6)
        self.violence_head = task_head(6)
        self.hate_head = task_head(3)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        
        # Passage par la couche partagée (768 -> 768)
        shared_out = self.shared_layer(pooled_output)
        
        # Les têtes reçoivent la sortie de la couche partagée
        return {
            'emotion': self.emotion_head(shared_out),
            'violence': self.violence_head(shared_out),
            'hate': self.hate_head(shared_out)
        }

app = FastAPI(title="Multi-Task NLP API")
MODEL_PATH = "models/final_model_projet4"
model = None
tokenizer = None

EMOTION_LABELS = ['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']
VIOLENCE_LABELS = ['Trad_practice', 'Physical', 'Economic', 'Emotional', 'Sexual', 'No_Violence']
HATE_LABELS = ['Hate Speech', 'Offensive', 'Neither']

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    try:
        checkpoint_path = f"{MODEL_PATH}/model.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Initialisation avec l'architecture corrigée
        model = MultiTaskClassifier(checkpoint['model_config']['model_name'])
        
        # Chargement des poids
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Tokenizer (Slow mode obligatoire pour spm.model)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
        
        logger.info("✅ ARCHITECTURE ET POIDS CHARGÉS AVEC SUCCÈS !")
    except Exception as e:
        logger.error(f"❌ Erreur critique : {e}")

@app.get("/")
def root():
    return {"status": "online", "model": "MultiTask DeBERTa v3 Corrected"}

@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "healthy", "is_model_loaded": model is not None}

@app.post("/predict", response_model=NLPResponse)
def predict(request: NLPRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    try:
        inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        
        with torch.no_grad():
            outputs = model(inputs['input_ids'], inputs['attention_mask'])
        
        res = {}
        for task, labels in [('emotion', EMOTION_LABELS), ('violence', VIOLENCE_LABELS), ('hate', HATE_LABELS)]:
            probs = torch.softmax(outputs[task], dim=-1)
            conf, idx = torch.max(probs, dim=-1)
            res[task] = TaskResult(label=labels[idx.item()], confidence=float(conf.item()))

        return NLPResponse(**res)
    except Exception as e:
        logger.error(f"Erreur prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))