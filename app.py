# ======================================================================
# HATE SPEECH DETECTION API - Flask
# ======================================================================

from flask import Flask, request, jsonify
import joblib
import re
import os
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Asset Initialization ──────────────────────────────────────────────
print("Loading production pipeline artifacts...")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
VEC_PATH   = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")

try:
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    print(f"✓ Model loaded: {type(model).__name__}")
    print(f"✓ Vectorizer loaded")
except Exception as e:
    print(f"❌ CRITICAL ERROR: {e}")
    raise RuntimeError(f"Pipeline assets missing or corrupted: {e}")

# ── Label Mappings ────────────────────────────────────────────────────
LABEL_NAMES = {
    0: "normal",
    1: "offensive",
    2: "hate_speech"
}

LABEL_DISPLAY = {
    0: "Normal",
    1: "Offensive",
    2: "Hate Speech"
}

# ── Keyword Rule Layer ────────────────────────────────────────────────
OFFENSIVE_KEYWORDS = [
    "idiot", "stupid", "moron", "dumb", "fool", "imbecile",
    "retard", "retarded", "dumbass", "jackass", "asshole",
    "bastard", "loser", "pathetic", "worthless", "scum",
    "shut up", "go to hell", "go die", "kill yourself",
    "you suck", "piece of shit", "piece of garbage"
]

# ── Safe Phrase Override ──────────────────────────────────────────────
# Short, common positive/neutral expressions the model misclassifies
# due to spurious word-offensive co-occurrence in the Twitter training data
# (e.g. "love" appears in 84% offensive tweets in Davidson because it
# co-occurs with profanity, not because "love" itself is offensive).
SAFE_PHRASES = [
    "i love you", "i love this", "have a nice day", "have a good day",
    "this is great", "this is amazing", "thank you", "good morning",
    "good night", "i like this", "this is beautiful", "well done",
    "congratulations", "happy birthday", "i appreciate you"
]

def safe_phrase_check(text):
    lowered = text.lower().strip()
    for phrase in SAFE_PHRASES:
        if phrase in lowered:
            return True
    return False

def keyword_check(text):
    """
    Returns (True, matched_keyword) if text contains a known offensive keyword.
    Returns (False, None) otherwise.
    """
    lowered = text.lower()
    for kw in OFFENSIVE_KEYWORDS:
        if kw in lowered:
            return True, kw
    return False, None

# ── Warning Level ─────────────────────────────────────────────────────
def get_warning_level(label_id, proba):
    if label_id == 2:
        return "high"
    if label_id == 1:
        return "high" if proba[2] > 0.2 else "medium"
    if label_id == 0 and (proba[1] + proba[2]) >= 0.35:
        return "low"
    return "none"

def scores_to_dict(proba):
    return {
        "normal":      round(float(proba[0]), 4),
        "offensive":   round(float(proba[1]), 4),
        "hate_speech": round(float(proba[2]), 4)
    }

# ── Text Preprocessing ────────────────────────────────────────────────
def clean_input(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-z0-9\s']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── Routes ────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status":       "online",
        "message":      "Hate Speech Detection API",
        "project":      "Monitoring Suspicious Discussions on Online Platforms Using Machine Learning",
        "authors":      "Fongang Ngnijiko Helene Grace & Fosso Temfack Mirenda Cassidy",
        "architecture": f"{type(model).__name__} (SMOTE Only) + Keyword Rule Layer + Safe Phrase Override",
        "endpoints": {
            "POST /predict":       "Single text evaluation",
            "POST /predict/batch": "Batch evaluation (max 100 entries)",
            "GET  /health":        "Pipeline health check"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":             "healthy",
        "model_loaded":       True,
        "model_type":         type(model).__name__,
        "classes_registered": list(LABEL_DISPLAY.values()),
        "rule_layer_active":  True,
        "keyword_count":      len(OFFENSIVE_KEYWORDS),
        "safe_phrase_count":  len(SAFE_PHRASES)
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({
            "error": "Missing text field",
            "usage": '{"text": "your text here"}'
        }), 400

    text = data["text"]
    if not text or len(text.strip()) == 0:
        return jsonify({"error": "Text field cannot be blank"}), 400

    cleaned = clean_input(text)

    print(f"DEBUG cleaned: '{cleaned}'")
    print(f"DEBUG safe_check: {safe_phrase_check(cleaned)}")

    if len(cleaned.strip()) == 0:
        return jsonify({
            "prediction":        "Normal",
            "label":             "normal",
            "label_id":          0,
            "is_suspicious":     False,
            "warning_level":     "none",
            "confidence_scores": {"normal": 1.0, "offensive": 0.0, "hate_speech": 0.0},
            "original_text":     text,
            "preprocessed_text": cleaned,
            "detection_method":  "fallback",
            "notes":             "No meaningful content after preprocessing"
        })

    # Get ML scores once, reused everywhere below
    vector      = vectorizer.transform([cleaned])
    ml_label_id = int(model.predict(vector)[0])
    proba       = model.predict_proba(vector)[0]

    # ── Step 0: Safe phrase override ──────────────────────
    if safe_phrase_check(cleaned):
        return jsonify({
            "prediction":        "Normal",
            "label":             "normal",
            "label_id":          0,
            "is_suspicious":     False,
            "warning_level":     "none",
            "confidence_scores": scores_to_dict(proba),
            "original_text":     text,
            "preprocessed_text": cleaned,
            "detection_method":  "safe_phrase_override"
        })

    # ── Step 1: Keyword rule layer ────────────────────────
    keyword_hit, matched_word = keyword_check(cleaned)

    if keyword_hit:
        return jsonify({
            "prediction":        "Offensive",
            "label":             "offensive",
            "label_id":          1,
            "is_suspicious":     True,
            "warning_level":     "medium",
            "confidence_scores": scores_to_dict(proba),
            "original_text":     text,
            "preprocessed_text": cleaned,
            "detection_method":  "rule_layer",
            "matched_keyword":   matched_word
        })

    # ── Step 2: ML model result ───────────────────────────
    is_suspicious = ml_label_id in [1, 2]
    warning_level = get_warning_level(ml_label_id, proba)

    return jsonify({
        "prediction":        LABEL_DISPLAY[ml_label_id],
        "label":             LABEL_NAMES[ml_label_id],
        "label_id":          ml_label_id,
        "is_suspicious":     is_suspicious,
        "warning_level":     warning_level,
        "confidence_scores": scores_to_dict(proba),
        "original_text":     text,
        "preprocessed_text": cleaned,
        "detection_method":  "ml_model"
    })

@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    data = request.get_json()

    if not data or "texts" not in data:
        return jsonify({
            "error": "Missing texts field",
            "usage": '{"texts": ["text1", "text2"]}'
        }), 400

    texts = data["texts"]
    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({"error": "texts must be a non-empty array"}), 400
    if len(texts) > 100:
        return jsonify({"error": "Batch limit exceeded. Maximum 100 rows per call"}), 400

    cleaned_texts  = [clean_input(str(t)) for t in texts]
    vectors        = vectorizer.transform(cleaned_texts)
    ml_predictions = model.predict(vectors)
    all_proba      = model.predict_proba(vectors)

    results = []
    for idx, orig_text in enumerate(texts):
        cleaned_version = cleaned_texts[idx]

        if len(cleaned_version.strip()) == 0:
            results.append({
                "prediction":    "Normal",
                "label":         "normal",
                "label_id":      0,
                "is_suspicious": False,
                "warning_level": "none",
                "original_text": orig_text
            })
            continue

        proba       = all_proba[idx]
        ml_label_id = int(ml_predictions[idx])

        # Safe phrase override
        if safe_phrase_check(cleaned_version):
            results.append({
                "prediction":        "Normal",
                "label":             "normal",
                "label_id":          0,
                "is_suspicious":     False,
                "warning_level":     "none",
                "confidence_scores": scores_to_dict(proba),
                "original_text":     orig_text,
                "preprocessed_text": cleaned_version,
                "detection_method":  "safe_phrase_override"
            })
            continue

        # Keyword rule layer
        keyword_hit, matched_word = keyword_check(cleaned_version)
        if keyword_hit:
            results.append({
                "prediction":        "Offensive",
                "label":             "offensive",
                "label_id":          1,
                "is_suspicious":     True,
                "warning_level":     "medium",
                "confidence_scores": scores_to_dict(proba),
                "original_text":     orig_text,
                "preprocessed_text": cleaned_version,
                "detection_method":  "rule_layer",
                "matched_keyword":   matched_word
            })
            continue

        is_suspicious = ml_label_id in [1, 2]
        warning_level = get_warning_level(ml_label_id, proba)

        results.append({
            "prediction":        LABEL_DISPLAY[ml_label_id],
            "label":             LABEL_NAMES[ml_label_id],
            "label_id":          ml_label_id,
            "is_suspicious":     is_suspicious,
            "warning_level":     warning_level,
            "confidence_scores": scores_to_dict(proba),
            "original_text":     orig_text,
            "preprocessed_text": cleaned_version,
            "detection_method":  "ml_model"
        })

    return jsonify({
        "total":            len(results),
        "suspicious_count": sum(1 for r in results if r["is_suspicious"]),
        "distribution": {
            "normal":      sum(1 for r in results if r["label_id"] == 0),
            "offensive":   sum(1 for r in results if r["label_id"] == 1),
            "hate_speech": sum(1 for r in results if r["label_id"] == 2)
        },
        "warning_summary": {
            "high":   sum(1 for r in results if r.get("warning_level") == "high"),
            "medium": sum(1 for r in results if r.get("warning_level") == "medium"),
            "low":    sum(1 for r in results if r.get("warning_level") == "low"),
            "none":   sum(1 for r in results if r.get("warning_level") == "none")
        },
        "results": results
    })

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("   MONITORING SERVICE OPERATIONAL")
    print(f"   Model: {type(model).__name__} (SMOTE Only)")
    print("   URL  : http://127.0.0.1:5000")
    print("=" * 65)
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)