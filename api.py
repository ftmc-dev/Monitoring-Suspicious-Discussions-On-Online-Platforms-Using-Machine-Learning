# ============================================================
# HATE SPEECH DETECTION API - FastAPI
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import re
import uvicorn
from typing import Dict, List

# Initialize FastAPI app
app = FastAPI(
    title="Hate Speech Detection API",
    description="API for detecting hate speech, offensive language, and normal content",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and vectorizer
print("Loading model and vectorizer...")
model = joblib.load("./models/best_model.pkl")
vectorizer = joblib.load("./data/processed/tfidf_vectorizer.pkl")
print("✓ Model and vectorizer loaded successfully")

LABEL_NAMES = {0: "Normal", 1: "Offensive", 2: "Hate Speech"}

# Define request schema
class TextRequest(BaseModel):
    text: str

class TextResponse(BaseModel):
    original_text: str
    preprocessed_text: str
    prediction: str
    label_id: int
    confidence: float
    probabilities: Dict[str, float]

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.strip()
    return text

# Define prediction function
def predict_text(text: str):

    processed_text = preprocess_text(text)

    vec = vectorizer.transform([processed_text])

    pred_id = int(model.predict(vec)[0])
    pred_label = LABEL_NAMES[pred_id]

    probabilities = model.predict_proba(vec)[0]

    confidence = round(float(max(probabilities) * 100), 2)

    prob_dict = {
        LABEL_NAMES[i]: round(float(prob * 100), 2)
        for i, prob in enumerate(probabilities)
    }

    return (
        pred_id,
        pred_label,
        processed_text,
        confidence,
        prob_dict
    )

# API Endpoints
@app.get("/")
def root():
    return {
        "message": "Hate Speech Detection API",
        "endpoints": {
            "/predict": "POST - Predict single text",
            "/health": "GET - Check API status"
        }
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=TextResponse)
def predict(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    (
       pred_id,
       pred_label,
       processed_text,
       confidence,
       probabilities
     ) = predict_text(request.text)

    return TextResponse(
      original_text=request.text,
      preprocessed_text=processed_text,
      prediction=pred_label,
      label_id=pred_id,
      confidence=confidence,
      probabilities=probabilities
      )

@app.post("/predict/batch")
def predict_batch(requests: List[TextRequest]):
    results = []
    for req in requests:
        pred_id, pred_label = predict_text(req.text)
        results.append({
            "text": req.text,
            "prediction": pred_label,
            "label_id": pred_id
        })
    return {"results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)