"""Analyze token lengths for Stage 1 training data to check if max_seq=2048 is enough."""
import polars as pl
import math
import numpy as np

df = pl.read_csv('data_upload/sft_merged_v1.csv')
pdf = df.to_pandas()

def _has_thinking(thinking):
    if thinking is None:
        return False
    if isinstance(thinking, float) and math.isnan(thinking):
        return False
    s = str(thinking).strip()
    return len(s) > 0 and s.lower() != 'nan'

PROMPT_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

def build_stage1_text(row):
    prompt = row["prompt"]
    answer = str(row["answer"])
    thinking = row.get("thinking", None)
    if _has_thinking(thinking):
        text = (
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n{str(thinking).strip()}\n</think>\n{answer}<|im_end|>"
        )
    else:
        text = (
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n<think></think>{answer}<|im_end|>"
        )
    return text

# Load tokenizer
from transformers import AutoTokenizer
import kagglehub
# Use local model path or just count chars as proxy
# Actually let's use the tokenizer
try:
    MODEL_PATH = "/Users/hastws/.cache/kagglehub/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    USE_TOKENIZER = True
    print("Using real tokenizer")
except Exception as e:
    USE_TOKENIZER = False
    print(f"No tokenizer available, using char-based estimation: {e}")

# Build all texts and measure lengths
print(f"\nTotal rows: {len(pdf)}")
print(f"  With thinking: {pdf['thinking'].apply(_has_thinking).sum()}")
print(f"  Answer-only: {(~pdf['thinking'].apply(_has_thinking)).sum()}")

lengths = []
types = []
for i, row in pdf.iterrows():
    text = build_stage1_text(row.to_dict())
    if USE_TOKENIZER:
        toks = tokenizer(text, add_special_tokens=False)
        length = len(toks['input_ids'])
    else:
        length = len(text) // 3  # rough estimate
    lengths.append(length)
    types.append('thinking' if _has_thinking(row['thinking']) else 'answer-only')

lengths = np.array(lengths)
types = np.array(types)

print(f"\n{'='*60}")
print(f"  TOKEN LENGTH DISTRIBUTION (Stage 1)")
print(f"{'='*60}")

for label, mask in [("ALL", np.ones(len(lengths), dtype=bool)),
                     ("thinking", types == 'thinking'),
                     ("answer-only", types == 'answer-only')]:
    subset = lengths[mask]
    print(f"\n  [{label}] n={len(subset)}")
    print(f"    min:    {subset.min()}")
    print(f"    p25:    {np.percentile(subset, 25):.0f}")
    print(f"    median: {np.percentile(subset, 50):.0f}")
    print(f"    mean:   {subset.mean():.0f}")
    print(f"    p75:    {np.percentile(subset, 75):.0f}")
    print(f"    p90:    {np.percentile(subset, 90):.0f}")
    print(f"    p95:    {np.percentile(subset, 95):.0f}")
    print(f"    p99:    {np.percentile(subset, 99):.0f}")
    print(f"    max:    {subset.max()}")

# Check truncation at various thresholds
print(f"\n{'='*60}")
print(f"  TRUNCATION ANALYSIS")
print(f"{'='*60}")
for threshold in [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096]:
    truncated = (lengths > threshold).sum()
    pct = truncated / len(lengths) * 100
    think_trunc = ((lengths > threshold) & (types == 'thinking')).sum()
    print(f"  max_seq={threshold:5d}: {truncated:5d} truncated ({pct:5.1f}%) — {think_trunc} thinking, {truncated-think_trunc} AO")

# Show the longest samples
print(f"\n{'='*60}")
print(f"  TOP 20 LONGEST SAMPLES")
print(f"{'='*60}")
sorted_idx = np.argsort(lengths)[::-1]
for rank, idx in enumerate(sorted_idx[:20]):
    row = pdf.iloc[idx]
    t = 'T' if _has_thinking(row['thinking']) else 'A'
    text = build_stage1_text(row.to_dict())
    prompt_preview = row['prompt'][:60].replace('\n', ' ')
    print(f"  #{rank+1:2d} [{t}] {lengths[idx]:5d} tok | {prompt_preview}...")
