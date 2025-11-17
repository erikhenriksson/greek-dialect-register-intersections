import json
import os
import re

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Set ALL cache directories to local disk to avoid quota issues
os.environ["TRANSFORMERS_CACHE"] = "./cache/transformers"
os.environ["HF_HOME"] = "./cache/huggingface"
os.environ["TORCH_HOME"] = "./cache/torch"
os.environ["TRITON_CACHE_DIR"] = "./cache/triton"
os.environ["XDG_CACHE_HOME"] = "./cache"
os.environ["HF_HUB_CACHE"] = "./cache/hub"
os.environ["SPACY_DATA"] = "./cache/spacy"

# Create cache directories
os.makedirs("./cache/transformers", exist_ok=True)
os.makedirs("./cache/huggingface", exist_ok=True)
os.makedirs("./cache/torch", exist_ok=True)
os.makedirs("./cache/triton", exist_ok=True)
os.makedirs("./cache/hub", exist_ok=True)
os.makedirs("./cache/spacy", exist_ok=True)

# Configuration
INPUT_FILE = "/scratch/project_2002026/data/hplt3_samples/ell_Grek_1M.jsonl"
OUTPUT_FILE = "labeled_sentences.jsonl"
MODEL_PATH = "./greek_dialect_model"
BATCH_SIZE = 128  # Increased from 32 for better GPU utilization
MAX_LENGTH = 256  # Reduced from 8192 - sentences are typically much shorter
REGISTER_THRESHOLD = 0.4

print("=" * 60)
print("Greek Dialect Sentence Labeling")
print("=" * 60)

# Performance optimizations
torch.set_float32_matmul_precision("high")  # Use TF32 on Ampere+ GPUs
print("Set matmul precision to 'high' for TF32 acceleration")

# Load the fine-tuned model
print(f"\nLoading model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Load model in bfloat16 for faster inference (2x speedup, half memory)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print(f"Model loaded in bfloat16 precision for faster inference")

# Get label mapping
id2label = model.config.id2label
print(f"Dialects: {list(id2label.values())}")


# Sentence splitting function using regex (much faster than spaCy)
def split_sentences(text):
    """Fast sentence splitting using regex. Returns sentences >= 50 characters."""
    # Split on common Greek sentence endings
    sentences = re.split(r"[.!;·?]\s+", text)
    # Filter: strip whitespace and keep only sentences with 50+ characters
    return [s.strip() for s in sentences if len(s.strip()) >= 50]


print("Using fast regex-based sentence splitting (min 50 chars)")
print()
print(f"Processing: {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print(f"Batch size: {BATCH_SIZE} (increased for better GPU utilization)")
print(f"Max sequence length: {MAX_LENGTH} tokens (optimized for sentences)")
print(f"Min sentence length: 50 characters (matches training data)")
print(f"Register threshold: {REGISTER_THRESHOLD}")
print(f"Device: {device}")
print()


def predict_dialects_batch(sentences):
    """Predict dialect probabilities for a batch of sentences"""
    if not sentences:
        return []

    # Tokenize batch
    encodings = tokenizer(
        sentences,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    )

    # Move to device
    encodings = {k: v.to(device) for k, v in encodings.items()}

    # Predict
    with torch.no_grad():
        outputs = model(**encodings)
        probs = torch.softmax(outputs.logits, dim=-1)

    # Convert to list of dicts
    results = []
    for prob_tensor in probs:
        prob_dict = {id2label[i]: float(prob_tensor[i]) for i in range(len(id2label))}
        results.append(prob_dict)

    return results


def predict_sentences_sorted_by_length(sentences):
    """
    Predict dialects for sentences, using length-based batching for efficiency.
    Returns predictions in original sentence order.
    """
    if not sentences:
        return []

    # Create list of (index, sentence, length) tuples
    indexed_sentences = [(i, sent, len(sent)) for i, sent in enumerate(sentences)]

    # Sort by length
    sorted_sentences = sorted(indexed_sentences, key=lambda x: x[2])

    # Extract just the sentences in sorted order
    sorted_texts = [sent for _, sent, _ in sorted_sentences]

    # Batch predict on sorted sentences
    all_predictions = []
    for i in range(0, len(sorted_texts), BATCH_SIZE):
        batch = sorted_texts[i : i + BATCH_SIZE]
        batch_preds = predict_dialects_batch(batch)
        all_predictions.extend(batch_preds)

    # Create (original_index, prediction) pairs
    predictions_with_indices = [
        (orig_idx, pred)
        for (orig_idx, _, _), pred in zip(sorted_sentences, all_predictions)
    ]

    # Sort back to original order
    predictions_with_indices.sort(key=lambda x: x[0])

    # Return just the predictions in original order
    return [pred for _, pred in predictions_with_indices]


# Process documents
doc_count = 0
sentence_count = 0
error_count = 0

with (
    open(INPUT_FILE, "r", encoding="utf-8") as fin,
    open(OUTPUT_FILE, "w", encoding="utf-8") as fout,
):
    for line in tqdm(fin, desc="Processing documents"):
        try:
            doc = json.loads(line)
            doc_count += 1

            # Extract text and registers
            text = doc.get("text", "")
            web_register_probs = doc.get("web-register", {})

            # Skip empty documents
            if not text.strip():
                continue

            # Get registers above threshold
            registers = [
                k for k, v in web_register_probs.items() if v >= REGISTER_THRESHOLD
            ]

            # Sentence segmentation with regex (fast) and filter >= 50 chars
            sentences = split_sentences(text)

            if not sentences:
                continue

            # Batch predict dialects with length-based sorting for efficiency
            dialect_predictions = predict_sentences_sorted_by_length(sentences)

            # Write results for each sentence
            for sent_id, (sentence_text, dialect_probs) in enumerate(
                zip(sentences, dialect_predictions)
            ):
                result = {
                    "doc_id": doc_count - 1,
                    "sent_id": sent_id,
                    "text": sentence_text,
                    "registers": registers,
                    "web_register_probs": web_register_probs,  # Keep full probs for backup
                    "dialect_probs": {k: round(v, 4) for k, v in dialect_probs.items()},
                }
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                sentence_count += 1

        except Exception as e:
            error_count += 1
            if error_count <= 10:  # Only print first 10 errors
                print(f"\nError processing document {doc_count}: {e}")
            continue

print("\n" + "=" * 60)
print("Processing Complete")
print("=" * 60)
print(f"Documents processed: {doc_count:,}")
print(f"Sentences labeled: {sentence_count:,}")
print(
    f"Average sentences per doc: {sentence_count / doc_count:.1f}"
    if doc_count > 0
    else "N/A"
)
print(f"Errors: {error_count}")
print(f"\nOutput saved to: {OUTPUT_FILE}")
