# ============================================
# SIMPLIFIED MIN-K% CONTAMINATION DETECTOR
# Based on Shi et al. (2024)
# ============================================

import os
os.environ["HF_HOME"] = "./hf_home"

from datetime import datetime
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================
# CONFIGURATION
# ============================================

HF_TOKEN = ""  # Insert your Hugging Face token here

SAMPLE_SIZES = [5000, 15000, 25000]
K_VALUE = 0.20  # Standard k=20% from the paper

DIALECT_FILES = {
    "Pontic": "dialects/final_pontic.txt",
    "Cretan": "dialects/Cretan_final.txt",
    "Cypriot": "dialects/final_cypriot.txt",
    "Ponticplus": "dialects/Pontic_final.txt",
    "Northern": "dialects/Northern_final.txt",
    "Tsakonian": "dialects/final_tsakonian.txt",
    "Grico": "dialects/Griko_final.txt",
    "Heptanesian": "dialects/Eptanisian_final.txt",
    "Maniot": "dialects/final_maniot.txt",
    "SMG": "dialects/SMG.txt",
}

OUTPUT_DIR = "contamination_results_simplified"
MAX_LENGTH = 512
STRIDE = 256

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================
# UTILITIES
# ============================================

def log(message):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


# ============================================
# DATA LOADING
# ============================================

def load_dialect_corpus(filepath, sample_size, seed=42):
    """Load and sample a dialect corpus"""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    
    words = text.split()
    if len(words) < sample_size:
        return None
    
    np.random.seed(seed)
    start_idx = np.random.randint(0, max(1, len(words) - sample_size))
    sampled_words = words[start_idx : start_idx + sample_size]
    
    return " ".join(sampled_words)


def load_all_dialects(sample_size):
    """Load all dialects at a given sample size"""
    results = {}
    log(f"Loading dialects at {sample_size:,} words...")
    
    for name, filepath in DIALECT_FILES.items():
        if not os.path.exists(filepath):
            log(f"  ⚠️  {name}: File not found - {filepath}")
            continue
        
        text = load_dialect_corpus(filepath, sample_size)
        if text:
            results[name] = text
            log(f"  ✅ {name}")
    
    return results


# ============================================
# MODEL LOADING
# ============================================

def load_model():
    """Load LLaMA 3 8B model"""
    log("Loading Meta-LLaMA 3 8B...")
    
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        token=HF_TOKEN,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B",
        token=HF_TOKEN
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    log("✅ Model loaded\n")
    
    return model, tokenizer


# ============================================
# CORE MIN-K% ALGORITHM
# ============================================

def get_token_log_probabilities(text, model, tokenizer):
    """
    Extract log probabilities for all tokens using sliding window.
    
    Returns:
        numpy array of log probabilities (one per predicted token)
    """
    encodings = tokenizer(text, return_tensors="pt", truncation=False)
    seq_len = encodings.input_ids.size(1)
    
    if seq_len == 0:
        return None
    
    all_log_probs = []
    prev_end_loc = 0
    
    for begin_loc in range(0, seq_len, STRIDE):
        end_loc = min(begin_loc + MAX_LENGTH, seq_len)
        trg_len = end_loc - prev_end_loc
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            
            # Get logits and compute log probabilities
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            # Calculate log probabilities for each token
            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            
            # Extract the log probability of the actual next token
            token_log_probs = log_probs.gather(
                dim=-1, 
                index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            
            # Only keep the target tokens
            mask = shift_labels != -100
            valid_log_probs = token_log_probs[mask].float().cpu().numpy()
            all_log_probs.extend(valid_log_probs.tolist())
        
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    
    return np.array(all_log_probs)


def calculate_min_k_prob(log_probs, k=K_VALUE):
    """
    Calculate Min-K% probability metric.
    
    Core algorithm from Shi et al. (2024):
    - Sort log probabilities (lowest = most surprising tokens)
    - Take bottom k% 
    - Average them
    
    Higher average = more confident on hard tokens = likely contaminated
    Lower average = less confident = likely clean
    
    Args:
        log_probs: array of log probabilities for each token
        k: proportion of lowest probability tokens to examine
    
    Returns:
        dict with min-k% statistics
    """
    if log_probs is None or len(log_probs) == 0:
        return None
    
    # Sort log probabilities (lowest first)
    sorted_log_probs = np.sort(log_probs)
    
    # Take bottom k%
    k_cutoff = max(1, int(len(sorted_log_probs) * k))
    bottom_k = sorted_log_probs[:k_cutoff]
    
    return {
        "min_k_mean": np.mean(bottom_k),
        "min_k_std": np.std(bottom_k),
        "min_k_median": np.median(bottom_k),
        "num_tokens": len(log_probs),
        "k_cutoff_tokens": k_cutoff,
    }


# ============================================
# CONTAMINATION ANALYSIS
# ============================================

def analyze_contamination(texts_dict, model, tokenizer):
    """
    Analyze contamination for all dialects.
    
    Returns DataFrame with contamination metrics.
    """
    log(f"Analyzing contamination (k={K_VALUE:.0%})...")
    
    results = []
    
    for dialect, text in tqdm(texts_dict.items(), desc="Processing dialects"):
        log_probs = get_token_log_probabilities(text, model, tokenizer)
        
        if log_probs is not None:
            stats = calculate_min_k_prob(log_probs)
            
            if stats:
                results.append({
                    "dialect": dialect,
                    "min_k_mean": stats["min_k_mean"],
                    "min_k_std": stats["min_k_std"],
                    "min_k_median": stats["min_k_median"],
                    "num_tokens": stats["num_tokens"],
                })
                
                log(f"  {dialect:15s}: {stats['min_k_mean']:.4f} ({stats['num_tokens']:,} tokens)")
    
    return pd.DataFrame(results)


def calculate_contamination_scores(df, reference="SMG"):
    """
    Calculate contamination scores relative to reference dialect (typically SMG).
    
    Adds:
    - distance_from_ref: Absolute difference from reference
    - normalized_distance: Distance as proportion of reference score
    """
    if reference not in df["dialect"].values:
        log(f"⚠️  Reference dialect '{reference}' not found")
        return df
    
    ref_score = df[df["dialect"] == reference]["min_k_mean"].values[0]
    
    df["distance_from_smg"] = abs(df["min_k_mean"] - ref_score)
    df["normalized_distance"] = df["distance_from_smg"] / abs(ref_score)
    
    # Sort by confidence (highest/least negative = most confident = most contaminated)
    df = df.sort_values("min_k_mean", ascending=False)
    
    return df


# ============================================
# MAIN ANALYSIS
# ============================================

def main():
    log("=" * 80)
    log("MIN-K% CONTAMINATION DETECTION (SIMPLIFIED)")
    log("=" * 80)
    log(f"Using k={K_VALUE:.0%} (standard from Shi et al. 2024)")
    log(f"Sample sizes: {SAMPLE_SIZES}")
    log("")
    
    # Load model once
    model, tokenizer = load_model()
    
    # Process each sample size
    all_results = []
    
    for sample_size in SAMPLE_SIZES:
        log("=" * 80)
        log(f"PROCESSING {sample_size:,} WORDS")
        log("=" * 80)
        
        # Load dialects
        texts = load_all_dialects(sample_size)
        if not texts:
            log(f"⚠️  No dialects loaded for {sample_size:,}w")
            continue
        
        # Analyze contamination
        results_df = analyze_contamination(texts, model, tokenizer)
        
        # Calculate scores relative to SMG
        results_df = calculate_contamination_scores(results_df, reference="SMG")
        
        # Add sample size column
        results_df["sample_size"] = sample_size
        
        # Store results
        all_results.append(results_df)
        
        # Print summary
        log("\n" + "─" * 80)
        log("CONTAMINATION SUMMARY")
        log("─" * 80)
        print(results_df.to_string(index=False))
        log("")
        
        # Save individual size results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{OUTPUT_DIR}/contamination_{sample_size}w_{timestamp}.csv"
        results_df.to_csv(filename, index=False)
        log(f"✅ Saved: {filename}\n")
    
    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_filename = f"{OUTPUT_DIR}/contamination_all_sizes_{timestamp}.csv"
        combined_df.to_csv(combined_filename, index=False)
        log(f"✅ Saved combined results: {combined_filename}")
    
    # Print interpretation guide
    log("\n" + "=" * 80)
    log("INTERPRETATION GUIDE")
    log("=" * 80)
    log("min_k_mean:")
    log("  Higher (less negative) = Model more confident = Likely contaminated")
    log("  Lower (more negative) = Model less confident = Likely clean")
    log("")
    log("distance_from_smg:")
    log("  ~0.00-0.10 = Similar to SMG contamination level")
    log("  ~0.10-0.30 = Moderately different from SMG")
    log("  >0.30 = Much less contaminated than SMG")
    log("")
    log("normalized_distance:")
    log("  Proportion of reference score (easier to compare across datasets)")
    log("=" * 80)


if __name__ == "__main__":
    main()