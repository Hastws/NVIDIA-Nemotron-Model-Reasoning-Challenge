"""
Strip narrative fluff from prompts and create clean mathematical prompts.
Then test with NVIDIA API to measure base model's raw ceiling.

Each type is really just:
- numeral: Arabic → Roman numeral (given examples). Pure pattern matching.
- gravity: d = 0.5*g*t^2, infer g from examples, compute d. Linear regression.
- unit_conv: y = k*x, infer k from examples, compute y. Linear regression.  
- cipher: Substitution cipher (letter mapping). Pattern: decode from examples.
- bit_ops: 8-bit binary transformation f(x). Pattern: infer function from I/O pairs.
- symbol: Symbol equation transformation. Pattern: infer rule from equations.
"""
import polars as pl
import json
import re

df = pl.read_csv('competition_data/train.csv')

def classify(prompt):
    p = prompt.lower()
    for t, kw in {'bit_ops': 'bit manipulation', 'gravity': 'gravitational',
                   'unit_conv': 'unit conversion', 'cipher': 'encryption',
                   'numeral': 'numeral system', 'symbol': 'transformation rules'}.items():
        if kw in p:
            return t
    return 'unknown'

df = df.with_columns(pl.col("prompt").map_elements(classify, return_dtype=pl.Utf8).alias("type"))

def strip_prompt(prompt, ptype):
    """Remove narrative fluff, return clean mathematical prompt."""
    lines = prompt.strip().split('\n')
    
    if ptype == 'numeral':
        # Core: "Convert Arabic to Roman numeral. Examples: ..."
        examples = [l.strip() for l in lines if '->' in l and 'Now' not in l]
        question_line = [l.strip() for l in lines if l.strip().startswith('Now')]
        q_num = re.search(r'number (\d+)', question_line[0]).group(1) if question_line else "?"
        return f"Convert the number to the same numeral system shown in the examples.\nExamples:\n" + \
               "\n".join(examples) + f"\nConvert: {q_num}"
    
    elif ptype == 'gravity':
        # Core: "Given d = 0.5*g*t^2, infer g from data, compute d for new t"
        examples = [l.strip() for l in lines if 'distance =' in l]
        question_line = [l.strip() for l in lines if l.strip().startswith('Now')]
        t_match = re.search(r't = ([\d.]+)s', question_line[0]) if question_line else None
        t_val = t_match.group(1) if t_match else "?"
        return f"Given d = 0.5 * g * t^2, infer g from the observations and compute d.\nObservations:\n" + \
               "\n".join(examples) + f"\nCompute d for t = {t_val}s"
    
    elif ptype == 'unit_conv':
        # Core: "y = f(x), infer f from examples, compute f(x_new)"
        examples = [l.strip() for l in lines if 'becomes' in l]
        question_line = [l.strip() for l in lines if l.strip().startswith('Now')]
        x_match = re.search(r'measurement: ([\d.]+)', question_line[0]) if question_line else None
        x_val = x_match.group(1) if x_match else "?"
        return f"A linear conversion maps input to output. Infer the rule from examples.\nExamples:\n" + \
               "\n".join(examples) + f"\nConvert: {x_val}"
    
    elif ptype == 'cipher':
        # Core: "Substitution cipher. Decrypt using letter mapping from examples."
        examples = [l.strip() for l in lines if '->' in l and 'Now' not in l]
        question_line = [l.strip() for l in lines if l.strip().startswith('Now')]
        cipher_text = question_line[0].replace('Now, decrypt the following text: ', '') if question_line else "?"
        return f"Each letter maps to another letter (substitution cipher). Decrypt using the examples.\nExamples:\n" + \
               "\n".join(examples) + f"\nDecrypt: {cipher_text}"
    
    elif ptype == 'bit_ops':
        # Core: "8-bit binary transform. Infer f from I/O examples."
        examples = [l.strip() for l in lines if '->' in l and 'Now' not in l and 'Here' not in l]
        question_line = [l.strip() for l in lines if l.strip().startswith('Now')]
        input_match = re.search(r'for: ([01]+)', question_line[0]) if question_line else None
        input_val = input_match.group(1) if input_match else "?"
        return f"An unknown function transforms 8-bit binary numbers. Infer the rule from examples.\nExamples:\n" + \
               "\n".join(examples) + f"\nCompute f({input_val})"
    
    elif ptype == 'symbol':
        # Core: "Symbol equation transform. Infer rule from examples."
        examples = [l.strip() for l in lines if '=' in l and 'Now' not in l and 'Below' not in l]
        question_line = [l.strip() for l in lines if l.strip().startswith('Now')]
        input_match = re.search(r'for: (.+)', question_line[0]) if question_line else None
        input_val = input_match.group(1).strip() if input_match else "?"
        return f"Transformation rules map input expressions to outputs. Infer the rule.\nExamples:\n" + \
               "\n".join(examples) + f"\nCompute: {input_val}"
    
    return prompt  # fallback

# Process all and save
results = []
for row in df.iter_rows(named=True):
    stripped = strip_prompt(row["prompt"], row["type"])
    results.append({
        "id": row["id"],
        "type": row["type"],
        "original_prompt": row["prompt"],
        "stripped_prompt": stripped,
        "answer": row["answer"],
    })

# Save
with open('competition_data/stripped_prompts.jsonl', 'w') as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print(f"Saved {len(results)} stripped prompts to competition_data/stripped_prompts.jsonl")

# Show examples
for t in ['numeral', 'gravity', 'unit_conv', 'cipher', 'bit_ops', 'symbol']:
    sample = [r for r in results if r['type'] == t][0]
    print(f"\n{'='*50}")
    print(f"TYPE: {t}")
    print(f"{'='*50}")
    print(f"STRIPPED PROMPT:\n{sample['stripped_prompt']}")
    print(f"\nANSWER: {sample['answer']}")
