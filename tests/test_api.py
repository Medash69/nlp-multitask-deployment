import sys
import os
import torch
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

# Ajout du chemin pour trouver 'app'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import app

client = TestClient(app)

def test_read_root():
    """Vérifie la racine /"""
    response = client.get("/")
    assert response.status_code == 200
    # On vérifie juste la présence de 'status' pour être flexible
    assert "status" in response.json()

def test_health_check():
    """Vérifie /health"""
    response = client.get("/health")
    # Si ton code renvoie 404, on vérifie si la route existe dans main.py
    assert response.status_code == 200
    assert response.json()["is_model_loaded"] is not None

@patch("app.main.tokenizer")
@patch("app.main.model")
def test_predict_endpoint(mock_model, mock_tokenizer):
    """Test de prédiction avec Mocks robustes"""
    
    # 1. Simuler le tokenizer
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]])
    }

    # 2. Simuler la sortie du modèle (Dictionnaire de logits)
    # ATTENTION : Les tailles doivent correspondre exactement à tes labels
    mock_model.return_value = {
        'emotion': torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]), # 6 classes
        'violence': torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]), # 6 classes
        'hate': torch.tensor([[1.0, 0.0, 0.0]])                     # 3 classes
    }

    payload = {"text": "test message"}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    res_json = response.json()
    assert "emotion" in res_json
    assert "label" in res_json["emotion"]

def test_predict_invalid_json():
    """Vérifie le comportement sur mauvais format"""
    response = client.post("/predict", json={"wrong_key": "hello"})
    assert response.status_code == 422