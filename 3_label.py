import json
import os

import spacy
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
BATCH_SIZE = 32
REGISTER_THRESHOLD = 0.4

print("=" * 60)
print("Greek Dialect Sentence Labeling")
print("=" * 60)

# Load the fine-tuned model
print(f"\nLoading model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Get label mapping
id2label = model.config.id2label
print(f"Dialects: {list(id2label.values())}")

# Load spaCy for sentence segmentation
print("\nLoading spaCy for Greek sentence segmentation...")
try:
    nlp = spacy.load("el_core_news_sm")
except:
    print("Greek model not found. Downloading...")
    os.system("python -m spacy download el_core_news_sm")
    nlp = spacy.load("el_core_news_sm")

# Make sure sentencizer is in the pipeline
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

# Disable other pipes for speed, but keep sentencizer
pipes_to_disable = [pipe for pipe in nlp.pipe_names if pipe != "sentencizer"]
if pipes_to_disable:
    nlp.disable_pipes(pipes_to_disable)

print(f"\nProcessing: {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Register threshold: {REGISTER_THRESHOLD}")
print(f"Device: {device}")
print()


def predict_dialects_batch(sentences):
    """Predict dialect probabilities for a batch of sentences"""
    if not sentences:
        return []

    # Tokenize batch
    encodings = tokenizer(
        sentences, truncation=True, max_length=8192, padding=True, return_tensors="pt"
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

            # Sentence segmentation
            doc_spacy = nlp(text)
            sentences = [
                sent.text.strip() for sent in doc_spacy.sents if sent.text.strip()
            ]

            if not sentences:
                continue

            # Batch predict dialects
            dialect_predictions = predict_dialects_batch(sentences)

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
                    "dialect_probs": dialect_probs,
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
