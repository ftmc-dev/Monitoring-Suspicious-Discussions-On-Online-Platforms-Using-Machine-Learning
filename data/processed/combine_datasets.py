import pandas as pd
import json
import os
from collections import Counter

print("="*60)
print("  COMBINING DATASETS")
print("="*60)

# ── Load your existing preprocessed Reddit data ──
print("\nLoading your Reddit data...")
reddit_df = pd.read_csv('preprocessed_data.csv', encoding='utf-8')
print(f"Reddit comments: {len(reddit_df)}")
print(f"Distribution: {Counter(reddit_df['label'])}")

# ── Load Davidson dataset ──
print("\nLoading Davidson dataset...")
davidson_raw = pd.read_csv('davidson_raw.csv')
print(f"Davidson total: {len(davidson_raw)}")

# Davidson class column:
# 0 = hate speech
# 1 = offensive language
# 2 = neither (normal)
print(f"Davidson distribution: {Counter(davidson_raw['class'])}")

# ── Same preprocessing function ──
import re

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
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    words = [simple_stem(w) for w in words]
    return ' '.join(words)

# ── Convert Davidson labels to match your system ──
# Davidson: 0=hate_speech, 1=offensive, 2=normal
# Your system: 0=normal, 1=offensive, 2=hate_speech
def convert_davidson_label(cls):
    mapping = {0: 2, 1: 1, 2: 0}
    return mapping[cls]

# ── Preprocess Davidson tweets ──
print("\nPreprocessing Davidson tweets...")
davidson_texts = []
davidson_labels = []

for _, row in davidson_raw.iterrows():
    clean = preprocess(row['tweet'])
    if len(clean.strip()) > 2:
        davidson_texts.append(clean)
        davidson_labels.append(convert_davidson_label(row['class']))

davidson_df = pd.DataFrame({
    'text': davidson_texts,
    'label': davidson_labels
})

print(f"Davidson after preprocessing: {len(davidson_df)}")
print(f"Davidson distribution: {Counter(davidson_df['label'])}")

# ── Combine both datasets ──
print("\nCombining datasets...")
combined_df = pd.concat([reddit_df, davidson_df], ignore_index=True)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nCombined dataset: {len(combined_df)} total comments")
counts = Counter(combined_df['label'])
print(f"  Normal (0):      {counts[0]}")
print(f"  Offensive (1):   {counts[1]}")
print(f"  Hate Speech (2): {counts[2]}")

# ── Save combined dataset ──
combined_df.to_csv('preprocessed_data_combined.csv', index=False, encoding='utf-8')
print(f"\nSaved to preprocessed_data_combined.csv")
print("Done!")