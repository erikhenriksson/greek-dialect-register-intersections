import argparse  # <-- ADDED
import itertools  # <-- ADDED
import json
import os
import re

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --- NEW: Argument Parsing ---
parser = argparse.ArgumentParser(
    description="Process a specific chunk of the Greek JSONL file."
)
parser.add_argument("chunk", type=int, help="The chunk number to process (1-10).")
args = parser.parse_args()
chunk_number = args.chunk

if not 1 <= chunk_number <= 10:
    print(f"Error: Chunk number must be between 1 and 10, but got {chunk_number}")
    exit(1)

# --- NEW: Chunk Calculation ---
CHUNK_SIZE = 100_000
# 0-indexed start line
start_line = (chunk_number - 1) * CHUNK_SIZE
# 0-indexed exclusive end line
end_line = chunk_number * CHUNK_SIZE

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
# --- MODIFIED: Output file is now dynamic ---
OUTPUT_FILE = f"labeled_sentences_chunk_{chunk_number}.jsonl"
MODEL_PATH = "./greek_dialect_model"
BATCH_SIZE = 128  # Increased from 32 for better GPU utilization
MAX_LENGTH = 512  # Reduced from 8192 - sentences are typically much shorter
REGISTER_THRESHOLD = 0.4

print("=" * 60)
print("Greek Dialect Sentence Labeling")
print("=" * 60)
# --- NEW: Print chunk info ---
print(f"TARGETING CHUNK {chunk_number}")
print(f"Processing lines: {start_line + 1} to {end_line}")
print(f"Outputting to: {OUTPUT_FILE}")
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
print(f"Output: {OUTPUT_FILE}")  # This will now show the chunked name
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

# --- MODIFIED: Main processing loop ---
with (
    open(INPUT_FILE, "r", encoding="utf-8") as fin,
    open(OUTPUT_FILE, "w", encoding="utf-8") as fout,
):
    # Calculate total lines for this specific chunk
    total_lines_in_chunk = end_line - start_line

    # Create an iterator that slices the file from start_line to end_line
    chunk_iterator = itertools.islice(fin, start_line, end_line)

    # Set up tqdm for the chunk
    progress_bar_desc = f"Chunk {chunk_number} (Lines {start_line + 1}-{end_line})"
    progress_bar = tqdm(
        chunk_iterator, desc=progress_bar_desc, total=total_lines_in_chunk
    )

    for line in progress_bar:  # Iterate over the sliced chunk
        try:
            doc = json.loads(line)
            doc_count += 1  # This will count from 1 to 100,000 for the chunk

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
                    # --- MODIFIED: doc_id is now relative to the whole file ---
                    "doc_id": start_line + doc_count - 1,
                    "sent_id": sent_id,
                    "text": sentence_text,
                    "registers": registers,
                    "web_register_probs": web_register_probs,  # Keep full probs for backup
                    "dialect_probs": dialect_probs,
                }
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                sentence_count += 1

        except Exception as e:
            error_count += 1
            if error_count <= 10:  # Only print first 10 errors
                # Provides more context for the error
                print(
                    f"\nError processing document {doc_count} in chunk (line approx {start_line + doc_count}): {e}"
                )
            continue

print("\n" + "=" * 60)
print(f"Processing Complete for Chunk {chunk_number}")
print("=" * 60)
print(f"Documents processed (in chunk): {doc_count:,}")
print(f"Sentences labeled (in chunk): {sentence_count:,}")
print(
    f"Average sentences per doc: {sentence_count / doc_count:.1f}"
    if doc_count > 0
    else "N/A"
)
print(f"Errors: {error_count}")
print(f"\nOutput saved to: {OUTPUT_FILE}")
