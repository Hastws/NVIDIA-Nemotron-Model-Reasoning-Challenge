#!/usr/bin/env python3
"""Analyze structure of each puzzle type."""
import os, polars as pl, re
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

df = pl.read_csv('competition_data/train.csv')

def classify(p):
    p = p.lower()
    if 'bit manipulation' in p or '8-bit binary' in p: return 'bit_ops'
    elif 'encrypt' in p or 'decrypt' in p: return 'cipher'
    elif 'gravitational' in p or 'falling distance' in p: return 'gravity'
    elif 'numeral system' in p: return 'numeral'
    elif 'transformation rules' in p: return 'symbol'
    elif 'unit conversion' in p or 'convert the following measurement' in p: return 'unit_conv'
    return 'unknown'

df = df.with_columns(pl.col('prompt').map_elements(classify, return_dtype=pl.Utf8).alias('qtype'))

for qtype in sorted(df['qtype'].unique().to_list()):
    subset = df.filter(pl.col('qtype') == qtype)
    prompts = subset['prompt'].to_list()
    answers = subset['answer'].to_list()
    
    # Analyze structure
    prompt_lens = [len(p) for p in prompts]
    answer_lens = [len(str(a)) for a in answers]
    
    # Count number of examples in each prompt
    example_counts = []
    for p in prompts:
        # Count input->output pairs or similar patterns
        arrows = len(re.findall(r'->', p))
        example_counts.append(arrows)
    
    print(f'=== {qtype} ({len(subset)} total) ===')
    print(f'  Prompt len: min={min(prompt_lens)}, max={max(prompt_lens)}, avg={sum(prompt_lens)//len(prompt_lens)}')
    print(f'  Answer len: min={min(answer_lens)}, max={max(answer_lens)}, avg={sum(answer_lens)//len(answer_lens)}')
    print(f'  Examples/prompt (arrows): min={min(example_counts)}, max={max(example_counts)}, avg={sum(example_counts)//len(example_counts)}')
    
    # Show first prompt structure
    p0 = prompts[0]
    print(f'\n  --- Example prompt (first 500 chars) ---')
    print(f'  {p0[:500]}')
    print(f'  --- Answer: {answers[0]} ---')
    print()

# Check: are all prompts within a type structurally identical (same template)?
print('\n=== STRUCTURAL UNIFORMITY CHECK ===')
for qtype in sorted(df['qtype'].unique().to_list()):
    subset = df.filter(pl.col('qtype') == qtype)
    prompts = subset['prompt'].to_list()
    
    # Extract "skeleton" by removing numbers/specific values
    skeletons = set()
    for p in prompts[:20]:  # sample 20
        # Replace numbers with <N>, specific words with <W>
        skeleton = re.sub(r'\d+\.?\d*', '<N>', p)
        skeleton = re.sub(r'[A-Z]{3,}', '<W>', skeleton)  # uppercase words
        skeletons.add(skeleton[:200])  # first 200 chars of skeleton
    
    print(f'{qtype}: {len(skeletons)} unique skeletons out of 20 samples')
    if len(skeletons) <= 3:
        print(f'  → HIGHLY UNIFORM template')
    else:
        print(f'  → VARIABLE templates')
