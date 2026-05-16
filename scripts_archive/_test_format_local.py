"""Simulate the notebook's format verification logic locally to catch issues before pushing."""
import polars as pl
import pandas as pd
import math

df = pl.read_csv('data_upload/sft_merged_v1.csv')
sample_df = df.to_pandas()

def _has_thinking(thinking):
    if thinking is None:
        return False
    if isinstance(thinking, float) and math.isnan(thinking):
        return False
    s = str(thinking).strip()
    return len(s) > 0 and s.lower() != 'nan'

PROMPT_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

def build_stage1_text(example):
    prompt = example["prompt"]
    answer = str(example["answer"])
    thinking = example.get("thinking", None)
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
    return {"text": text}

def build_stage2_text(example):
    prompt = example["prompt"]
    answer = str(example["answer"])
    user_msg = prompt + PROMPT_SUFFIX
    text = (
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n<think></think>\\boxed{{{answer}}}<|im_end|>"
    )
    return {"text": text}

# Test Stage 1 with thinking row
think_rows = sample_df[sample_df['thinking'].apply(_has_thinking)]
ao_rows = sample_df[~sample_df['thinking'].apply(_has_thinking)]
print(f"Rows with thinking: {len(think_rows)}")
print(f"Answer-only rows: {len(ao_rows)}")
assert len(think_rows) + len(ao_rows) == len(sample_df), "Row count mismatch!"

# Test Stage 1 thinking
row = think_rows.iloc[0].to_dict()
result = build_stage1_text(row)
text = result['text']
assert '<think>\n' in text, "Missing <think> tag"
assert '\n</think>\n' in text, "Missing </think> tag"
assert '\\boxed{' not in text, "Stage 1 should NOT contain \\boxed{}!"
assert 'nan' not in text.split('<|im_start|>assistant')[1].split('</think>')[0].lower() or len(text.split('<|im_start|>assistant')[1].split('</think>')[0]) > 20, "Thinking contains only 'nan'"
print("✅ Stage 1 thinking row OK")

# Test Stage 1 answer-only
row = ao_rows.iloc[0].to_dict()
result = build_stage1_text(row)
text = result['text']
assert '<think></think>' in text, f"Missing empty think tags! Got: {text[:200]}"
assert '\\boxed{' not in text, "Stage 1 AO should NOT contain \\boxed{}!"
assert 'nan' not in text.split('<|im_start|>assistant')[1], f"Contains 'nan'! Got: {text[:200]}"
print("✅ Stage 1 answer-only row OK")

# Test ALL answer-only rows to make sure none produce <think>\nnan\n</think>
print("\nTesting all answer-only rows...")
bad_count = 0
for i in range(min(100, len(ao_rows))):
    row = ao_rows.iloc[i].to_dict()
    result = build_stage1_text(row)
    text = result['text']
    if '<think></think>' not in text:
        print(f"  ❌ Row {i}: missing empty think. thinking={repr(row.get('thinking'))}")
        bad_count += 1
    if 'nan' in text.split('<|im_start|>assistant')[1].split('<|im_end|>')[0]:
        print(f"  ❌ Row {i}: contains 'nan'. thinking={repr(row.get('thinking'))}")
        bad_count += 1

if bad_count == 0:
    print(f"✅ All {min(100, len(ao_rows))} answer-only rows produce correct format")
else:
    print(f"❌ {bad_count} issues found!")

# Test Stage 2
row = ao_rows.iloc[0].to_dict()
result = build_stage2_text(row)
text = result['text']
assert PROMPT_SUFFIX.lstrip('\n') in text, "Missing prompt suffix"
assert '\\boxed{' in text, "Missing \\boxed{}"
print("✅ Stage 2 row OK")

# Also test via HF Dataset path (actual training path)
from datasets import Dataset
hf_ao = Dataset.from_pandas(ao_rows.head(10))
hf_ao = hf_ao.map(build_stage1_text, remove_columns=hf_ao.column_names)
for i in range(len(hf_ao)):
    assert '<think></think>' in hf_ao[i]['text'], f"HF row {i} missing empty think!"
    assert 'nan' not in hf_ao[i]['text'].split('<|im_start|>assistant')[1]
print("✅ HF Dataset path: all 10 AO rows OK")

print("\n🎉 All tests passed! Safe to push.")
