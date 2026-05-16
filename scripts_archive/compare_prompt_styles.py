"""
Compare original vs instructional prompts on 1 sample per type.
Shows both prompts and full API responses side by side.
"""
import os, json, re, math
from openai import OpenAI

API_BASE = "https://integrate.api.nvidia.com/v1"
MODEL = "nvidia/nemotron-3-nano-30b-a3b"
METRIC_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY")
client = OpenAI(base_url=API_BASE, api_key=api_key)

# --- Build instructional prompts per type ---
def build_instructional_prompt(original, ptype):
    """Replace narrative with clear type identification + step-by-step instructions."""
    lines = original.strip().split('\n')
    
    if ptype == 'numeral':
        # Extract examples and question
        examples = [l.strip() for l in lines if '->' in l and 'Now' not in l]
        question_line = [l.strip() for l in lines if l.strip().startswith('Now')]
        q_num = re.search(r'number (\d+)', question_line[0]).group(1) if question_line else "?"
        
        return f"""This is an Arabic-to-Roman numeral conversion problem.

Steps:
1. Verify the examples below are standard Arabic → Roman numeral mappings
2. Convert the given number to Roman numerals using standard rules (I=1, V=5, X=10, L=50, C=100, subtractive notation for 4,9,40,90)
3. Output the result directly

Examples:
{chr(10).join(examples)}

Convert: {q_num}
Output the Roman numeral directly."""

    elif ptype == 'gravity':
        examples = [l.strip() for l in lines if 'distance =' in l]
        question_line = [l.strip() for l in lines if l.strip().startswith('Now')]
        t_match = re.search(r't = ([\d.]+)s', question_line[0]) if question_line else None
        t_val = t_match.group(1) if t_match else "?"
        
        return f"""This is a physics free-fall problem with an unknown gravitational constant g.

Formula: d = 0.5 * g * t²

Steps:
1. From each observation, compute g = 2*d / t²
2. Average all computed g values
3. Compute d = 0.5 * g_avg * t_query² for the query time
4. Round to 2 decimal places

Observations:
{chr(10).join(examples)}

Compute d for t = {t_val}s. Output the numeric result directly (2 decimal places)."""

    elif ptype == 'unit_conv':
        examples = [l.strip() for l in lines if 'becomes' in l]
        question_line = [l.strip() for l in lines if l.strip().startswith('Now')]
        x_match = re.search(r'measurement: ([\d.]+)', question_line[0]) if question_line else None
        x_val = x_match.group(1) if x_match else "?"
        
        return f"""This is a linear unit conversion problem: output = factor * input.

Steps:
1. From each example, compute factor = output / input
2. Average all computed factors
3. Compute result = factor_avg * query_input
4. Round to 2 decimal places

Examples:
{chr(10).join(examples)}

Convert: {x_val}. Output the numeric result directly (2 decimal places)."""

    elif ptype == 'cipher':
        examples = [l.strip() for l in lines if '->' in l and 'Now' not in l]
        question_line = [l.strip() for l in lines if l.strip().startswith('Now')]
        cipher_text = question_line[0].replace('Now, decrypt the following text: ', '') if question_line else "?"
        
        return f"""This is a substitution cipher problem. Each letter in the encrypted text maps to exactly one letter in the plaintext (a monoalphabetic cipher).

Steps:
1. From the example pairs, build the letter mapping: encrypted_char → plaintext_char
2. Apply the mapping to decrypt the query text
3. Spaces and word boundaries are preserved

Examples (encrypted -> plaintext):
{chr(10).join(examples)}

Decrypt: {cipher_text}
Output the decrypted plaintext directly."""

    elif ptype == 'bit_ops':
        examples = [l.strip() for l in lines if '->' in l and 'Now' not in l and 'Here' not in l]
        question_line = [l.strip() for l in lines if l.strip().startswith('Now')]
        input_match = re.search(r'for: ([01]+)', question_line[0]) if question_line else None
        input_val = input_match.group(1) if input_match else "?"
        
        return f"""This is an 8-bit binary transformation problem. An unknown bitwise function f maps 8-bit inputs to 8-bit outputs.

Steps:
1. Analyze the input→output pairs to identify the transformation rule
2. The rule may involve: bit shifts, rotations, XOR, AND, OR, NOT, or combinations
3. Test your hypothesis against ALL examples to verify
4. Apply the verified rule to the query input

Examples (input -> output):
{chr(10).join(examples)}

Compute f({input_val}). Output the 8-bit binary result directly."""

    elif ptype == 'symbol':
        examples = [l.strip() for l in lines if '=' in l and 'Now' not in l and 'Below' not in l]
        question_line = [l.strip() for l in lines if l.strip().startswith('Now')]
        input_match = re.search(r'for: (.+)', question_line[0]) if question_line else None
        input_val = input_match.group(1).strip() if input_match else "?"
        
        return f"""This is a symbolic equation transformation problem. A rule maps input expressions to output expressions.

Steps:
1. Analyze each example to identify what changes between input and output
2. Look for patterns: character substitution, deletion, reordering, or operator-based rules
3. The rule applies consistently across all examples
4. Apply the rule to the query expression

Examples:
{chr(10).join(examples)}

Compute the result for: {input_val}
Output the result directly."""

    return original


def call_api(prompt, enable_thinking=True):
    messages = [{"role": "user", "content": prompt + METRIC_SUFFIX}]
    resp = client.chat.completions.create(
        model=MODEL, messages=messages,
        temperature=0.0, top_p=1.0, max_tokens=7680,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
        timeout=180,
    )
    choice = resp.choices[0]
    content = choice.message.content or ""
    return content, choice.finish_reason


def extract_boxed(text):
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    for m in reversed(matches):
        if m.strip():
            return m.strip()
    return None


def answers_match(pred, gold):
    if pred is None: return False
    pred, gold = pred.strip(), gold.strip()
    if pred.lower() == gold.lower(): return True
    try: return math.isclose(float(pred), float(gold), rel_tol=1e-2, abs_tol=1e-5)
    except: return False


# Load data
with open('competition_data/stripped_prompts.jsonl') as f:
    data = [json.loads(l) for l in f]

# 1 sample per type
for t in ['numeral', 'gravity', 'unit_conv', 'cipher', 'bit_ops', 'symbol']:
    sample = next(r for r in data if r['type'] == t)
    
    original_prompt = sample['original_prompt']
    instructional_prompt = build_instructional_prompt(original_prompt, t)
    gold = sample['answer']
    
    print(f"\n{'='*80}")
    print(f"TYPE: {t}  |  ID: {sample['id']}  |  GOLD: {gold}")
    print(f"{'='*80}")
    
    # Show instructional prompt
    print(f"\n--- INSTRUCTIONAL PROMPT ---")
    print(instructional_prompt)
    
    # Call API for both
    print(f"\n--- CALLING API (original) ---")
    resp_orig, fin_orig = call_api(original_prompt)
    pred_orig = extract_boxed(resp_orig)
    match_orig = answers_match(pred_orig, gold)
    
    print(f"  finish: {fin_orig}")
    print(f"  length: {len(resp_orig)} chars")
    print(f"  predicted: {pred_orig}")
    print(f"  correct: {'✓' if match_orig else '✗'}")
    # Show last 500 chars (the answer part after thinking)
    if '</think>' in resp_orig:
        after_think = resp_orig.split('</think>')[-1].strip()
        print(f"  [after </think>]: {after_think[:300]}")
    
    print(f"\n--- CALLING API (instructional) ---")
    resp_inst, fin_inst = call_api(instructional_prompt)
    pred_inst = extract_boxed(resp_inst)
    match_inst = answers_match(pred_inst, gold)
    
    print(f"  finish: {fin_inst}")
    print(f"  length: {len(resp_inst)} chars")
    print(f"  predicted: {pred_inst}")
    print(f"  correct: {'✓' if match_inst else '✗'}")
    if '</think>' in resp_inst:
        after_think = resp_inst.split('</think>')[-1].strip()
        print(f"  [after </think>]: {after_think[:300]}")
    
    print(f"\n  COMPARISON: original={'✓' if match_orig else '✗'}  instructional={'✓' if match_inst else '✗'}")
