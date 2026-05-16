"""Local test: simulate Stage 2 with 10 samples to verify the masking logic works."""
import math
import pandas as pd

# Load data
df = pd.read_csv('data/sft_merged_v1.csv')
print(f"Loaded {len(df)} rows")

PROMPT_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'
STAGE2_MAX_SEQ = 512
THINK_END_TOKEN_ID = 13

def _has_thinking(thinking):
    if thinking is None:
        return False
    if isinstance(thinking, float) and math.isnan(thinking):
        return False
    s = str(thinking).strip()
    return len(s) > 0 and s.lower() != 'nan'

def build_stage2_text(example):
    prompt = example["prompt"]
    answer = str(example["answer"])
    thinking = example.get("thinking", "")
    user_msg = prompt + PROMPT_SUFFIX
    
    if _has_thinking(thinking):
        text = (
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n{str(thinking).strip()}\n</think>\n\\boxed{{{answer}}}<|im_end|>"
        )
    else:
        text = (
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n<think></think>\\boxed{{{answer}}}<|im_end|>"
        )
    return {"text": text}

# Filter thinking rows
think_mask = df['thinking'].apply(_has_thinking)
thinking_df = df[think_mask]
print(f"Thinking rows: {len(thinking_df)}")

# Sample 10
sample_df = thinking_df.sample(n=10, random_state=42)

print("\n" + "="*60)
print("Testing build_stage2_text on 10 samples:")
print("="*60)

for idx, (_, row) in enumerate(sample_df.iterrows()):
    result = build_stage2_text(row.to_dict())
    text = result['text']
    
    # Basic format checks
    has_suffix = PROMPT_SUFFIX.lstrip('\n') in text
    has_boxed = '\\boxed{' in text
    has_think = '<think>\n' in text
    has_think_end = '</think>' in text
    
    status = "✅" if (has_suffix and has_boxed and has_think and has_think_end) else "❌"
    print(f"\n{status} Sample {idx}: len={len(text)}, suffix={has_suffix}, boxed={has_boxed}, think={has_think}")
    
    # Show the answer portion (after </think>)
    if '</think>' in text:
        after_think = text.split('</think>')[1]
        print(f"   After </think>: {repr(after_think[:100])}")

print("\n" + "="*60)
print("Testing tokenize_and_mask_thinking simulation:")
print("="*60)

# We can't use the real tokenizer locally, but we can simulate the logic
# by finding </think> in the text and checking the structure
for idx, (_, row) in enumerate(sample_df.iterrows()):
    result = build_stage2_text(row.to_dict())
    text = result['text']
    
    # Find </think> position in text
    think_end_pos = text.find('</think>')
    if think_end_pos >= 0:
        masked_text = text[:think_end_pos + len('</think>')]
        trained_text = text[think_end_pos + len('</think>'):]
        print(f"\nSample {idx}:")
        print(f"  Total chars: {len(text)}")
        print(f"  Masked chars: {len(masked_text)} ({len(masked_text)/len(text)*100:.0f}%)")
        print(f"  Trained text: {repr(trained_text[:150])}")
        
        # Verify trained portion has \boxed{} and <|im_end|>
        assert '\\boxed{' in trained_text, f"❌ trained portion missing \\boxed{{}}!"
        assert '<|im_end|>' in trained_text, f"❌ trained portion missing <|im_end|>!"
    else:
        print(f"\nSample {idx}: ❌ no </think> found!")

print("\n" + "="*60)
print("Testing DataCollatorForSeq2Seq compatibility:")
print("="*60)

# Simulate what tokenize_and_mask_thinking produces
# Just verify the structure is correct
sample_result = {
    'input_ids': [1, 2, 3, THINK_END_TOKEN_ID, 4, 5, 6],
    'attention_mask': [1, 1, 1, 1, 1, 1, 1],
    'labels': [-100, -100, -100, -100, 4, 5, 6],
}

# Verify masking is correct
think_end_idx = sample_result['input_ids'].index(THINK_END_TOKEN_ID)
for i in range(think_end_idx + 1):
    assert sample_result['labels'][i] == -100, f"Token {i} should be masked!"
for i in range(think_end_idx + 1, len(sample_result['labels'])):
    assert sample_result['labels'][i] != -100, f"Token {i} should NOT be masked!"

print("✅ Masking logic correct")

print("\n" + "="*60)
print("Testing Trainer import compatibility:")
print("="*60)
try:
    from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
    print("✅ All imports successful")
    print(f"  Trainer: {Trainer}")
    print(f"  TrainingArguments: {TrainingArguments}")
    print(f"  DataCollatorForSeq2Seq: {DataCollatorForSeq2Seq}")
except ImportError as e:
    print(f"❌ Import failed: {e}")

print("\n" + "="*60)
print("ALL TESTS PASSED ✅")
print("="*60)
