import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertTokenizerFast, 
    DistilBertForSequenceClassification, 
    Trainer, 
    TrainingArguments
)

print("=" * 70)
print("PHASE 2: TRAINING CONTEXT-AWARE DISTILBERT ENGINE")
print("=" * 70)

# 1. Target the exact clean combined CSV your notebook saves
processed_data_path = "./data/processed/combined_cleaned_data.csv"

if os.path.exists(processed_data_path):
    print(f"✅ Found dataset at: {processed_data_path}")
    df = pd.read_csv(processed_data_path)
else:
    # Backup pathway in case you run the script directly from a different root folder
    print("⚠️ Processed folder not detected directly. Checking current directory...")
    try:
        df = pd.read_csv("combined_cleaned_data.csv")
    except Exception as e:
        print(f"❌ Error: Could not locate your clean data. {str(e)}")
        print("Please ensure your previous notebook ran successfully through the cleaning steps.")
        exit()

# Reconfirm your columns and drop any blank rows
# Your dataset uses columns: 'text' (the clean string) and 'label' (0, 1, 2)
df = df[['text', 'label']].dropna()
print(f"📊 Dataset successfully prepared. Total rows to process: {df.shape[0]}")

# 2. Extract Text and Labels
X = df['text'].astype(str).tolist()
y = df['label'].astype(int).tolist()

# Train/Validation Split (80% / 20%) maintaining exact class distribution ratios
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Load the pre-trained DistilBERT Tokenizer
print("\n🤖 Downloading/Loading DistilBERT Tokenizer base configuration...")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Tokenize text data (This encodes words into dense continuous context vectors)
print("🔤 Tokenizing dataset arrays...")
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(X_val, truncation=True, padding=True, max_length=128)

# 4. Wrap data into PyTorch Datasets
class ModerationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ModerationDataset(train_encodings, y_train)
val_dataset = ModerationDataset(val_encodings, y_val)

# 5. Load the Sequence Classification Model Configuration for 3 Classes (0, 1, 2)
print("🧠 Instantiating DistilBERT Architecture...")
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', 
    num_labels=3
)

# 6. Set Deep Learning Optimization Arguments
training_args = TrainingArguments(
    output_dir='./results',          # Checkpoint log repository
    num_train_epochs=2,              # 2 loops through your dataset is optimal for fine-tuning
    per_device_train_batch_size=16,  # Batch constraints for system stability
    per_device_eval_batch_size=64,
    warmup_steps=300,
    weight_decay=0.01,               # Regularization to minimize training overfitting
    logging_dir='./logs',
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

# 7. Initialize Trainer Framework
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 8. Run Fine-Tuning Execution
print("\n🚀 Starting Training Loop. DistilBERT is now mapping your dataset context...")
trainer.train()

# 9. Commit the Output Model Folder
output_dir = "./advanced_moderation_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"\n✅ SUCCESS: Context-aware model engine saved to folder: '{output_dir}'")