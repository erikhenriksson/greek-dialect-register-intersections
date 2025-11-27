import os
import re

os.environ["HF_HOME"] = "./model_cache"

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# File paths
FILES = {
    "standard": [
        "DATA/v1/clean_file_SMG_part1.txt",
        "DATA/v1/clean_file_SMG_part2.txt",
        "DATA/v1/clean_file_SMG_part3.txt",
    ],
    "cretan": ["DATA/v2/cretan.txt"],
    "cypriot": ["DATA/v2/cypriot.txt"],
    "northern": ["DATA/v2/nothern.txt"],
    "pontic": ["DATA/v2/pontic.txt"],
}

texts = []
labels = []

print("Loading data...")

# Load Standard Greek (v1 files)
for filepath in FILES["standard"]:
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.endswith(";Greek"):
                line = line[:-6].strip()
            if line:
                texts.append(line)
                labels.append("standard")

# Load Cretan (needs sentence splitting)
for filepath in FILES["cretan"]:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        # Remove metadata lines
        content = re.sub(r"---.*?---", "", content)
        # Split into sentences (simple split on common punctuation)
        sentences = re.split(r"[.!;]+", content)
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                texts.append(sentence)
                labels.append("cretan")

# Load other dialects (already good)
for dialect in ["cypriot", "northern", "pontic"]:
    for filepath in FILES[dialect]:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
                    labels.append(dialect)

print(f"\nTotal samples: {len(texts):,}")
for dialect in ["standard", "cretan", "cypriot", "northern", "pontic"]:
    count = sum(1 for l in labels if l == dialect)
    print(f"  {dialect}: {count:,}")

# Create label mapping
unique_labels = sorted(set(labels))
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}
label_ids = [label2id[label] for label in labels]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    texts, label_ids, test_size=0.2, random_state=42, stratify=label_ids
)

print(f"\nTrain: {len(X_train):,}, Test: {len(X_test):,}")

# Load model
model_name = "answerdotai/ModernBERT-large"
print(f"\nLoading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id,
    torch_dtype=torch.bfloat16,
)


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=8192,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx]),
        }


train_dataset = TextDataset(X_train, y_train, tokenizer)
test_dataset = TextDataset(X_test, y_test, tokenizer)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    tf32=True,
    report_to="none",
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

print("\nTraining...")
trainer.train()

print("\nEvaluating...")
predictions = trainer.predict(test_dataset)
y_pred = predictions.predictions.argmax(axis=1)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=unique_labels))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Order:", unique_labels)

print("\nSaving model...")
trainer.save_model("./greek_dialect_model")
tokenizer.save_pretrained("./greek_dialect_model")
print("Done!")
