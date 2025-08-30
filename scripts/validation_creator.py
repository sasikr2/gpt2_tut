"""
OOD Validation Dataset Builder (Streaming Optimized)
Creates an Out-of-Distribution (OOD) validation mix for GPT-style training.
All large datasets use streaming mode to avoid full downloads.

Token Budget:
- C4-en                                   20M tokens
- WikiText-103                             0.25M tokens
- Reasoning mix (HellaSwag+PIQA+ARC)        1M tokens

Saves: ./validation_data/ood_val_000000.npy
"""

import os
import numpy as np
import tiktoken
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

# -------------------------------
# CONFIG
# -------------------------------
local_dir = "validation_data"
os.makedirs(local_dir, exist_ok=True)

token_budgets = {
    "c4": int(20e6),
    "wiki": int(10e6),
    "openwebtext2": int(10e6),
    "books3": int(5e6),
    "reasoning": int(5e6),
}

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token
def tokenize(doc):
    """Tokenizes a single document and returns a numpy array of uint16 tokens."""
    tokens = [eot] # The special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "Token dictionary too large for uint16"
    return tokens_np.astype(np.uint16)

def sample_and_tokenize_streaming(dataset_iter, token_budget, desc):
    """
    (Improved for performance)
    Samples from a streaming dataset until the token budget is reached.
    Uses a Python list for efficient token collection.
    """
    collected_tokens = []
    pbar = tqdm(total=token_budget, unit="tok", desc=f"Tokenizing {desc}")
    
    for doc in dataset_iter:
        # Tokenize returns a numpy array, convert to list to extend efficiently
        toks = tokenize(doc).tolist()
        
        collected_tokens.extend(toks)
        pbar.update(len(toks))
        
        if len(collected_tokens) >= token_budget:
            break
            
    pbar.close()
    
    # Truncate to the exact budget and convert to a NumPy array only once
    final_tokens = np.array(collected_tokens[:token_budget], dtype=np.uint16)
    return final_tokens

# -------------------------------
# LOAD & PROCESS DATASETS
# -------------------------------
final_tokens = []

# 1. C4-en (streaming)
print("Processing C4...")
c4_stream = load_dataset("allenai/c4", "en", split="validation", streaming=True)
final_tokens.append(sample_and_tokenize_streaming(c4_stream, token_budgets["c4"], "C4-en"))

# 2. WikiText-103 (small enough to load fully, using 'validation' split)
print("\nProcessing WikiText-103...")
wiki_ds = load_dataset("wikitext", "wikitext-103-v1", split="validation", streaming=True)
final_tokens.append(sample_and_tokenize_streaming(iter(wiki_ds), token_budgets["wiki"], "WikiText-103"))

# # 3. OpenWebText2 (streaming)
# print("\nProcessing OpenWebText2...")
# owt2_stream = load_dataset("imka/openwebtext2", split="train", streaming=True)
# final_tokens.append(sample_and_tokenize_streaming(owt2_stream, token_budgets["openwebtext2"], "OpenWebText2"))

# 4. Books3 (streaming if available)
print("\nAttempting to process Books3...")
try:
    books3_stream = load_dataset("the_pile_books3", split="train", streaming=True)
    final_tokens.append(sample_and_tokenize_streaming(books3_stream, token_budgets["books3"], "Books3"))
except Exception as e:
    print(f"Could not process Books3 (this is common): {e}")

# 5. Reasoning mix (small enough to load fully)
print("\nProcessing Reasoning Mix...")
# Load the original datasets first to easily get their column names
hella_ds = load_dataset("hellaswag", split="validation")
piqa_ds = load_dataset("baber/piqa", split="validation")
arc_ds = load_dataset("ai2_arc", "ARC-Easy", split="validation")

# Now map them, removing the original columns
hella = hella_ds.map(
    lambda x: {"text": x["ctx"] + " " + x["endings"][int(x["label"])]},
    remove_columns=hella_ds.column_names
)
piqa = piqa_ds.map(
    lambda x: {"text": x["goal"] + " " + x["sol1"] + " " + x["sol2"]},
    remove_columns=piqa_ds.column_names
)
arc = arc_ds.map(
    lambda x: {"text": x["question"] + " " + " ".join(x["choices"]["text"])},
    remove_columns=arc_ds.column_names
)

# Concatenation will now succeed as all datasets have identical features
reasoning_ds = concatenate_datasets([hella, piqa, arc])
final_tokens.append(sample_and_tokenize_streaming(iter(reasoning_ds), token_budgets["reasoning"], "Reasoning Mix")) 
# -------------------------------
# MERGE, SHUFFLE & SAVE
# -------------------------------
print("\nMerging, shuffling, and saving final dataset...")
all_tokens = np.concatenate(final_tokens)
print(f"Total tokens before shuffle: {len(all_tokens):,}")

# Use a reproducible random number generator
rng = np.random.default_rng(seed=42)
rng.shuffle(all_tokens)

filename = os.path.join(local_dir, "ood_val_000000.npy")
np.save(filename, all_tokens)

print(f"\n✅ Successfully saved OOD validation set to {filename} with {len(all_tokens):,} tokens.")


# validation_data/ood_val_000000.npy with 21,219,075 tokens.