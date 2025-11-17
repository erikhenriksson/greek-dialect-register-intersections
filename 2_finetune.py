import json
import os
import re

# Set ALL cache directories to local
os.environ["TRANSFORMERS_CACHE"] = "./model_cache"
os.environ["HF_HOME"] = "./model_cache"
os.environ["TRITON_CACHE_DIR"] = "./triton_cache"
os.environ["TORCH_HOME"] = "./torch_cache"
os.environ["XDG_CACHE_HOME"] = "./cache"

# Create cache directories
os.makedirs("./model_cache", exist_ok=True)
os.makedirs("./triton_cache", exist_ok=True)
os.makedirs("./torch_cache", exist_ok=True)
os.makedirs("./cache", exist_ok=True)

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

# ========== PERFORMANCE OPTIMIZATIONS ==========
# Use TF32 on Ampere+ GPUs for faster matrix multiplication
torch.set_float32_matmul_precision("high")
print("Set matmul precision to 'high' for TF32 acceleration")
# ================================================


def count_greek_chars(text):
    """Count Greek characters in text (including accented characters)"""
    # Greek Unicode ranges: \u0370-\u03FF (Greek and Coptic), \u1F00-\u1FFF (Greek Extended)
    greek_pattern = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]")
    return len(greek_pattern.findall(text))


def is_valid_line(text, min_greek_chars=50):
    """Check if line has at least min_greek_chars Greek characters"""
    return count_greek_chars(text) >= min_greek_chars


def split_into_sentences(text):
    """Simple sentence splitter for Greek text"""
    # Split on common sentence endings followed by space or newline
    sentences = re.split(r"[.!;·]\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def split_long_text_into_chunks(text, target_chunks=2, min_chunk_size=500):
    """Split long text into chunks. First tries sentences, then falls back to character-based splitting"""
    sentences = split_into_sentences(text)

    # If we got multiple sentences, use sentence-based splitting
    if len(sentences) > target_chunks:
        sentences_per_chunk = max(1, len(sentences) // target_chunks)
        chunks = []
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk = " ".join(sentences[i : i + sentences_per_chunk])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    # Fall back to character-based splitting for texts without sentence boundaries
    text_len = len(text)
    if text_len < min_chunk_size * target_chunks:
        # Text too short to split meaningfully
        return [text]

    chunk_size = text_len // target_chunks
    chunks = []

    for i in range(target_chunks):
        start = i * chunk_size
        # Last chunk gets the remainder
        if i == target_chunks - 1:
            end = text_len
        else:
            end = (i + 1) * chunk_size
            # Try to find a space near the boundary to avoid splitting words
            # Look within 50 characters before the boundary
            search_start = max(start, end - 50)
            space_pos = text.rfind(" ", search_start, end)
            if space_pos != -1:
                end = space_pos

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

    return chunks


# Load data from dialect files
dialect_files = {
    "cretan": "DATA/v2/cretan.txt",
    "cypriot": "DATA/v2/cypriot.txt",
    "northern": "DATA/v2/nothern.txt",  # Note: typo in original filename
    "pontic": "DATA/v2/pontic.txt",
}

# Standard Greek files
standard_greek_files = [
    "DATA/v1/clean_file_SMG_part1.txt",
    "DATA/v1/clean_file_SMG_part2.txt",
    "DATA/v1/clean_file_SMG_part3.txt",
]

texts = []
labels = []

# Load Standard Greek data
print("Loading Standard Greek data...")
smg_count = 0
smg_skipped = 0
for filepath in standard_greek_files:
    print(f"  Loading {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Remove ";Greek" suffix
            if line.endswith(";Greek"):
                line = line[:-6].strip()

            # Skip header lines or metadata
            if (
                ";lang" in line
                or line.startswith("article;")
                or line.startswith("title ")
            ):
                smg_skipped += 1
                continue

            # Check if valid (at least 50 Greek chars)
            if is_valid_line(line):
                texts.append(line)
                labels.append("standard")
                smg_count += 1
            else:
                smg_skipped += 1

print(f"  Loaded {smg_count} documents (skipped {smg_skipped})")

# Load dialect data
for dialect_name, filepath in dialect_files.items():
    print(f"\nLoading {dialect_name} from {filepath}...")
    dialect_count = 0
    dialect_skipped = 0

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

        # Special handling for Cretan: split long documents into more chunks
        if dialect_name == "cretan":
            lines = []
            for line in content.split("\n"):
                line = line.strip()
                # Skip metadata lines like "--- cleaned_vakches.txt ---"
                if line.startswith("---") or "---" in line:
                    dialect_skipped += 1
                    continue
                if is_valid_line(line):
                    lines.append(line)
                elif line:
                    dialect_skipped += 1

            print(f"  Found {len(lines)} valid lines before splitting")

            # Calculate how many chunks we need per document to get more data
            target_total = 2000  # Adjust this number to get more/less Cretan data
            chunks_per_doc = max(2, target_total // len(lines)) if len(lines) > 0 else 4

            print(
                f"  Splitting long texts into ~{chunks_per_doc} chunks each to reach ~{target_total} samples"
            )

            # Split each long document into multiple chunks
            for line in lines:
                chunks = split_long_text_into_chunks(line, target_chunks=chunks_per_doc)
                for chunk in chunks:
                    if is_valid_line(chunk):
                        texts.append(chunk)
                        labels.append(dialect_name)
                        dialect_count += 1

        # Special handling for Pontic: split very long texts to get ~1000 samples
        elif dialect_name == "pontic":
            lines = []
            for line in content.split("\n"):
                line = line.strip()
                if is_valid_line(line):
                    lines.append(line)
                elif line:
                    dialect_skipped += 1

            print(f"  Found {len(lines)} valid lines before splitting")

            # Calculate how many chunks we need per document on average
            target_total = 1000
            chunks_per_doc = max(2, target_total // len(lines)) if len(lines) > 0 else 2

            print(
                f"  Splitting long texts into ~{chunks_per_doc} chunks each to reach ~{target_total} samples"
            )

            for line in lines:
                # Split long texts into chunks
                chunks = split_long_text_into_chunks(line, target_chunks=chunks_per_doc)
                for chunk in chunks:
                    if is_valid_line(chunk):
                        texts.append(chunk)
                        labels.append(dialect_name)
                        dialect_count += 1

        else:
            # For other dialects: line by line
            for line in content.split("\n"):
                line = line.strip()
                if is_valid_line(line):
                    texts.append(line)
                    labels.append(dialect_name)
                    dialect_count += 1
                else:
                    if line:  # Only count non-empty skipped lines
                        dialect_skipped += 1

    print(f"  Loaded {dialect_count} documents (skipped {dialect_skipped})")

print(f"\n{'=' * 60}")
print(f"Total documents loaded (before downsampling): {len(texts)}")
print(f"{'=' * 60}")
print(f"Documents per dialect:")
all_dialects = ["standard", "cretan", "cypriot", "northern", "pontic"]
for dialect in all_dialects:
    count = sum(1 for l in labels if l == dialect)
    percentage = (count / len(texts) * 100) if len(texts) > 0 else 0
    print(f"  {dialect:12s}: {count:8,} ({percentage:5.2f}%)")

# Downsample Standard Greek to balance dataset
import random

random.seed(42)

print(f"\n{'=' * 60}")
print("Downsampling Standard Greek to 100,000 samples...")
print(f"{'=' * 60}")

# Find indices of standard greek samples
standard_indices = [i for i, label in enumerate(labels) if label == "standard"]
other_indices = [i for i, label in enumerate(labels) if label != "standard"]

# Sample 100,000 from standard greek
if len(standard_indices) > 100000:
    sampled_standard_indices = random.sample(standard_indices, 100000)
    print(f"Downsampled Standard Greek from {len(standard_indices):,} to 100,000")
else:
    sampled_standard_indices = standard_indices
    print(
        f"Standard Greek has {len(standard_indices):,} samples (no downsampling needed)"
    )

# Combine sampled standard with all other dialects
selected_indices = sampled_standard_indices + other_indices
random.shuffle(selected_indices)

# Create new texts and labels with downsampled data
texts = [texts[i] for i in selected_indices]
labels = [labels[i] for i in selected_indices]

print(f"\nTotal documents after downsampling: {len(texts)}")
print(f"Documents per dialect:")
for dialect in all_dialects:
    count = sum(1 for l in labels if l == dialect)
    percentage = (count / len(texts) * 100) if len(texts) > 0 else 0
    print(f"  {dialect:12s}: {count:8,} ({percentage:5.2f}%)")

# Create label mapping
unique_labels = sorted(set(labels))
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}
label_ids = [label2id[label] for label in labels]

print(f"\nLabel mapping: {label2id}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    texts, label_ids, test_size=0.2, random_state=42, stratify=label_ids
)

print(f"\nTrain size: {len(X_train):,}, Test size: {len(X_test):,}")

# Load ModernBERT
model_name = "answerdotai/ModernBERT-large"
print(f"\nLoading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ========== PERFORMANCE OPTIMIZATION: Load model in bfloat16 ==========
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for 2x speedup and half memory
)
print(f"Model loaded in bfloat16 precision for faster training")
# ======================================================================


# Dataset class
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

# Training arguments
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
    bf16=True,  # ========== CHANGED: Use bfloat16 instead of fp16 ==========
    report_to="none",
)


# Metric function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}


# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

print("\n" + "=" * 60)
print("Training ModernBERT on Greek Dialects...")
print("=" * 60)
trainer.train()

# Evaluate
print("\n" + "=" * 60)
print("Evaluating on test set...")
print("=" * 60)
predictions = trainer.predict(test_dataset)
y_pred = predictions.predictions.argmax(axis=1)

print("\n" + "=" * 60)
print("MODERNBERT GREEK DIALECT CLASSIFICATION RESULTS")
print("=" * 60)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print()

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=unique_labels))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print()
print("Dialect order:", unique_labels)

# Save the model
print("\n" + "=" * 60)
print("Saving model to ./greek_dialect_model")
print("=" * 60)
trainer.save_model("./greek_dialect_model")
tokenizer.save_pretrained("./greek_dialect_model")
print("Done!")
