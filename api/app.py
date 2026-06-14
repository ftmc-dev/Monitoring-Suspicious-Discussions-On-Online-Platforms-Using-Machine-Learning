from flask import Flask, request, jsonify
import pickle
import re
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Load saved model and vectorizer ──
print("Loading model and vectorizer...")

with open('models/svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

print("Model loaded successfully!")

# ── Same stopwords from preprocess.py ──
STOPWORDS = set([
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up','down',
    'in','out','on','off','over','under','again','further','then','once','here',
    'there','when','where','why','how','all','both','each','few','more','most',
    'other','some','such','no','nor','not','only','own','same','so','than',
    'too','very','s','t','can','will','just','don','should','now','d','ll',
    'm','o','re','ve','y','ain','aren','couldn','didn','doesn','hadn','hasn',
    'haven','isn','ma','mightn','mustn','needn','shan','shouldn','wasn','weren',
    'won','wouldn'
])

def simple_stem(word):
    suffixes = ['ing','tion','ness','ment','able','ible','ed','er','ly','es','s']
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    words = [simple_stem(w) for w in words]
    return ' '.join(words)

label_map = {
    0: 'normal',
    1: 'offensive',
    2: 'hate_speech'
}

# ── API Endpoints ──

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Hate Speech Detection API',
        'project': 'Monitoring Suspicious Discussions on Online Platforms Using ML',
        'authors': 'Fongang Ngnijiko Helene Grace & Fosso Temfack Mirenda Cassidy',
        'endpoints': {
            'POST /predict': 'Send text for hate speech detection',
            'GET /health': 'Check API status'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'running',
        'model': 'SVM Linear',
        'vectorizer': 'TF-IDF 5000 features'
    })

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON input
    data = request.get_json()

    # Validate input
    if not data or 'text' not in data:
        return jsonify({
            'error': 'Missing text field',
            'usage': 'Send JSON with text field: {"text": "your comment here"}'
        }), 400

    text = data['text']

    if not text or len(text.strip()) == 0:
        return jsonify({
            'error': 'Text field is empty'
        }), 400

    # Preprocess
    clean_text = preprocess(text)

    if len(clean_text.strip()) == 0:
        return jsonify({
            'label': 'normal',
            'label_code': 0,
            'message': 'Text too short or no meaningful content after preprocessing',
            'original_text': text
        })

    # Vectorize and predict
    text_vector = vectorizer.transform([clean_text])
    prediction = model.predict(text_vector)[0]
    label = label_map[prediction]

    # Determine if suspicious
    is_suspicious = prediction in [1, 2]

    return jsonify({
        'label': label,
        'label_code': int(prediction),
        'is_suspicious': is_suspicious,
        'original_text': text,
        'preprocessed_text': clean_text
    })

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    data = request.get_json()

    if not data or 'texts' not in data:
        return jsonify({
            'error': 'Missing texts field',
            'usage': 'Send JSON with texts array: {"texts": ["comment1", "comment2"]}'
        }), 400

    texts = data['texts']

    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({'error': 'texts must be a non-empty array'}), 400

    if len(texts) > 100:
        return jsonify({'error': 'Maximum 100 texts per batch request'}), 400

    results = []
    for text in texts:
        clean_text = preprocess(str(text))
        if len(clean_text.strip()) == 0:
            results.append({
                'label': 'normal',
                'label_code': 0,
                'is_suspicious': False,
                'original_text': text
            })
        else:
            text_vector = vectorizer.transform([clean_text])
            prediction = model.predict(text_vector)[0]
            label = label_map[prediction]
            results.append({
                'label': label,
                'label_code': int(prediction),
                'is_suspicious': prediction in [1, 2],
                'original_text': text
            })

    suspicious_count = sum(1 for r in results if r['is_suspicious'])

    return jsonify({
        'total': len(results),
        'suspicious_count': suspicious_count,
        'results': results
    })

if __name__ == '__main__':
    print("\n" + "="*55)
    print("  HATE SPEECH DETECTION API STARTING...")
    print("="*55)
    print("  URL: http://127.0.0.1:5000")
    print("  Endpoints:")
    print("    GET  /         - API info")
    print("    GET  /health   - Health check")
    print("    POST /predict  - Single prediction")
    print("    POST /predict/batch - Batch prediction")
    print("="*55 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)