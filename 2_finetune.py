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
torch.set_float32_matmul_precision("high")
print("Set matmul precision to 'high' for TF32 acceleration")
# ================================================


def count_greek_chars(text):
    """Count Greek characters in text (including accented characters)"""
    greek_pattern = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]")
    return len(greek_pattern.findall(text))


def remove_punctuation(text):
    """Remove ALL punctuation from text to avoid spurious correlations"""
    # Remove common punctuation marks but keep spaces
    punctuation = r'[.!;·?,\-—–"\'(){}[\]:«»"""' "]"
    text = re.sub(punctuation, "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_valid_chunk(text, min_greek_chars=50):
    """Check if chunk has at least min_greek_chars Greek characters"""
    return count_greek_chars(text) >= min_greek_chars


def split_into_uniform_chunks(text, target_chunk_size=400, min_chunk_size=200):
    """
    Split text into uniform-sized chunks by character count.
    This ensures all dialects have similar-length samples regardless of punctuation.

    Args:
        text: Input text
        target_chunk_size: Target characters per chunk (default 400)
        min_chunk_size: Minimum characters for last chunk (default 200)
    """
    text = text.strip()
    text_len = len(text)

    if text_len < min_chunk_size:
        return [text] if text else []

    chunks = []
    start = 0

    while start < text_len:
        end = start + target_chunk_size

        # If this would be the last chunk and it's too small, extend previous chunk
        if end >= text_len:
            chunk = text[start:].strip()
            if chunk and len(chunk) >= min_chunk_size:
                chunks.append(chunk)
            elif chunks and chunk:
                # Merge small last chunk with previous chunk
                chunks[-1] = chunks[-1] + " " + chunk
            elif chunk:
                # First and only chunk that's smaller than min_chunk_size
                chunks.append(chunk)
            break

        # Try to find a space near the boundary to avoid splitting words
        search_start = max(start, end - 50)
        space_pos = text.rfind(" ", search_start, end + 50)

        if space_pos != -1 and space_pos > start:
            end = space_pos

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end

    return chunks


# Load data from dialect files
dialect_files = {
    "cretan": "DATA/v2/cretan.txt",
    "cypriot": "DATA/v2/cypriot.txt",
    "northern": "DATA/v2/nothern.txt",
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

print("=" * 60)
print("IMPORTANT: Normalizing all data to prevent spurious correlations")
print("=" * 60)
print("- Removing ALL punctuation from ALL dialects")
print("- Creating uniform-length chunks (~400 chars)")
print("- This ensures model learns linguistic features, not formatting")
print("=" * 60)

# Load Standard Greek data
print("\nLoading Standard Greek data...")
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

            # Remove punctuation
            line = remove_punctuation(line)

            # Check if valid
            if is_valid_chunk(line):
                # Split into uniform chunks if text is long
                chunks = split_into_uniform_chunks(line)
                for chunk in chunks:
                    if is_valid_chunk(chunk):
                        texts.append(chunk)
                        labels.append("standard")
                        smg_count += 1
            else:
                smg_skipped += 1

print(f"  Loaded {smg_count} chunks (skipped {smg_skipped})")

# Load dialect data
for dialect_name, filepath in dialect_files.items():
    print(f"\nLoading {dialect_name} from {filepath}...")
    dialect_count = 0
    dialect_skipped = 0

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

        # Process line by line for all dialects
        for line in content.split("\n"):
            line = line.strip()

            # Skip metadata lines
            if line.startswith("---") or "---" in line:
                dialect_skipped += 1
                continue

            if not line:
                continue

            # Remove punctuation (critical for normalization!)
            line = remove_punctuation(line)

            if not line:
                dialect_skipped += 1
                continue

            # Split into uniform chunks
            chunks = split_into_uniform_chunks(line)

            for chunk in chunks:
                if is_valid_chunk(chunk):
                    texts.append(chunk)
                    labels.append(dialect_name)
                    dialect_count += 1
                else:
                    dialect_skipped += 1

    print(f"  Loaded {dialect_count} chunks (skipped {dialect_skipped})")

print(f"\n{'=' * 60}")
print(f"Total chunks loaded (before balancing): {len(texts)}")
print(f"{'=' * 60}")
print(f"Chunks per dialect:")
all_dialects = ["standard", "cretan", "cypriot", "northern", "pontic"]
for dialect in all_dialects:
    count = sum(1 for l in labels if l == dialect)
    percentage = (count / len(texts) * 100) if len(texts) > 0 else 0
    print(f"  {dialect:12s}: {count:8,} ({percentage:5.2f}%)")

# ========== BALANCE DATASET ==========
import random

random.seed(42)

print(f"\n{'=' * 60}")
print("Balancing dataset...")
print(f"{'=' * 60}")

# Separate by dialect
dialect_indices = {dialect: [] for dialect in all_dialects}
for i, label in enumerate(labels):
    dialect_indices[label].append(i)

# Print original counts
print("\nOriginal counts:")
for dialect in all_dialects:
    count = len(dialect_indices[dialect])
    print(f"  {dialect:12s}: {count:8,}")

# Define target counts for each dialect
TARGET_STANDARD = 200_000  # Majority class
TARGET_DIALECT = 20_000  # Same for all dialects

targets = {
    "standard": TARGET_STANDARD,
    "cypriot": TARGET_DIALECT,
    "cretan": TARGET_DIALECT,
    "northern": TARGET_DIALECT,
    "pontic": TARGET_DIALECT,
}

print(f"\nTarget counts:")
for dialect, target in targets.items():
    print(f"  {dialect:12s}: {target:8,}")

# Balance each dialect
balanced_indices = []

for dialect in all_dialects:
    indices = dialect_indices[dialect]
    target = targets[dialect]
    current = len(indices)

    if current >= target:
        # Downsample
        sampled = random.sample(indices, target)
        print(f"\n{dialect}: Downsampling from {current:,} to {target:,}")
    else:
        # Upsample by sampling with replacement
        sampled = random.choices(indices, k=target)
        print(
            f"\n{dialect}: Upsampling from {current:,} to {target:,} (with replacement)"
        )

    balanced_indices.extend(sampled)

# Shuffle all indices
random.shuffle(balanced_indices)

# Create balanced texts and labels
texts = [texts[i] for i in balanced_indices]
labels = [labels[i] for i in balanced_indices]

print(f"\n{'=' * 60}")
print(f"Balanced dataset:")
print(f"{'=' * 60}")
print(f"Total chunks: {len(texts):,}")
print(f"\nChunks per dialect:")
for dialect in all_dialects:
    count = sum(1 for l in labels if l == dialect)
    percentage = (count / len(texts) * 100) if len(texts) > 0 else 0
    print(f"  {dialect:12s}: {count:8,} ({percentage:5.2f}%)")

# Sample a few examples to verify normalization
print(f"\n{'=' * 60}")
print("Sample chunks (first 200 chars):")
print(f"{'=' * 60}")
for dialect in all_dialects:
    dialect_samples = [texts[i] for i, l in enumerate(labels) if l == dialect]
    if dialect_samples:
        sample = dialect_samples[0][:200]
        print(f"\n{dialect}:")
        print(f"  {sample}...")
        print(f"  Length: {len(dialect_samples[0])} chars")

# ========================================

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

# Load model in bfloat16
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id,
    torch_dtype=torch.bfloat16,
)
print(f"Model loaded in bfloat16 precision for faster training")


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
    bf16=True,
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
