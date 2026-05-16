"""
Compare original vs v2 prompts: 
- bit_ops/symbol: short step-by-step
- numeral/gravity/unit_conv/cipher: direct answer, no thinking
"""
import os, json, re, math
from openai import OpenAI

API_BASE = "https://integrate.api.nvidia.com/v1"
MODEL = "nvidia/nemotron-3-nano-30b-a3b"
METRIC_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY")
client = OpenAI(base_url=API_BASE, api_key=api_key)


def build_v2_prompt(original, ptype):
    """V2: 4 types direct, 2 types short step-by-step."""
    lines = original.strip().split('\n')

    if ptype == 'numeral':
        examples = [l.strip() for l in lines if '->' in l and 'Now' not in l]
        question_line = [l.strip() for l in lines if l.strip().startswith('Now')]
        q_num = re.search(r'number (\d+)', question_line[0]).group(1) if question_line else "?"
        return f"Arabic to Roman numeral. Examples:\n" + \
               "\n".join(examples) + f"\nConvert: {q_num}\nDirectly output the answer."

    elif ptype == 'gravity':
        examples = [l.strip() for l in lines if 'distance =' in l]
        question_line = [l.strip() for l in lines if l.strip().startswith('Now')]
        t_match = re.search(r't = ([\d.]+)s', question_line[0]) if question_line else None
        t_val = t_match.group(1) if t_match else "?"
        return f"d = 0.5*g*t^2. Compute g from data, then compute d. Round to 2 decimals.\n" + \
               "\n".join(examples) + f"\nt = {t_val}s, d = ?\nDirectly output the number."

    elif ptype == 'unit_conv':
        examples = [l.strip() for l in lines if 'becomes' in l]
        question_line = [l.strip() for l in lines if l.strip().startswith('Now')]
        x_match = re.search(r'measurement: ([\d.]+)', question_line[0]) if question_line else None
        x_val = x_match.group(1) if x_match else "?"
        return f"Linear conversion: output = factor * input. Compute factor, then convert. Round to 2 decimals.\n" + \
               "\n".join(examples) + f"\nConvert: {x_val}\nDirectly output the number."

    elif ptype == 'cipher':
        examples = [l.strip() for l in lines if '->' in l and 'Now' not in l]
        question_line = [l.strip() for l in lines if l.strip().startswith('Now')]
        cipher_text = question_line[0].replace('Now, decrypt the following text: ', '') if question_line else "?"
        return f"Substitution cipher. Build letter mapping from examples, then decrypt.\n" + \
               "\n".join(examples) + f"\nDecrypt: {cipher_text}\nDirectly output the plaintext."

    elif ptype == 'bit_ops':
        examples = [l.strip() for l in lines if '->' in l and 'Now' not in l and 'Here' not in l]
        question_line = [l.strip() for l in lines if l.strip().startswith('Now')]
        input_match = re.search(r'for: ([01]+)', question_line[0]) if question_line else None
        input_val = input_match.group(1) if input_match else "?"
        return f"8-bit binary transform. Find the rule from examples.\n" + \
               "Step 1: Check each bit position — is output bit determined by input bits via shift, rotate, XOR, AND, OR, NOT?\n" + \
               "Step 2: Verify rule against ALL examples.\n" + \
               "Step 3: Apply to query.\n" + \
               "\n".join(examples) + f"\nf({input_val}) = ?"

    elif ptype == 'symbol':
        examples = [l.strip() for l in lines if '=' in l and 'Now' not in l and 'Below' not in l]
        question_line = [l.strip() for l in lines if l.strip().startswith('Now')]
        input_match = re.search(r'for: (.+)', question_line[0]) if question_line else None
        input_val = input_match.group(1).strip() if input_match else "?"
        return f"Symbol transformation. Find the rule from examples.\n" + \
               "Step 1: Compare input/output — what stays, what changes, what's removed?\n" + \
               "Step 2: Check if operator (+,-,*) affects the transformation.\n" + \
               "Step 3: Apply rule to query.\n" + \
               "\n".join(examples) + f"\n{input_val} = ?"

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

for t in ['numeral', 'gravity', 'unit_conv', 'cipher', 'bit_ops', 'symbol']:
    sample = next(r for r in data if r['type'] == t)

    original_prompt = sample['original_prompt']
    v2_prompt = build_v2_prompt(original_prompt, t)
    gold = sample['answer']

    print(f"\n{'='*80}")
    print(f"TYPE: {t}  |  ID: {sample['id']}  |  GOLD: {gold}")
    print(f"{'='*80}")

    print(f"\n[V2 PROMPT]")
    print(v2_prompt)

    # Original with thinking
    print(f"\n[ORIGINAL + thinking=True]")
    resp, fin = call_api(original_prompt, enable_thinking=True)
    pred = extract_boxed(resp)
    ok = answers_match(pred, gold)
    think_part = resp.split('</think>')[0] if '</think>' in resp else ""
    after_think = resp.split('</think>')[-1].strip() if '</think>' in resp else resp.strip()
    print(f"  finish={fin}  len={len(resp)}  think_len={len(think_part)}  pred={pred}  {'✓' if ok else '✗'}")
    if after_think:
        print(f"  answer_part: {after_think[:200]}")

    # V2 with thinking
    print(f"\n[V2 + thinking=True]")
    resp2, fin2 = call_api(v2_prompt, enable_thinking=True)
    pred2 = extract_boxed(resp2)
    ok2 = answers_match(pred2, gold)
    think_part2 = resp2.split('</think>')[0] if '</think>' in resp2 else ""
    after_think2 = resp2.split('</think>')[-1].strip() if '</think>' in resp2 else resp2.strip()
    print(f"  finish={fin2}  len={len(resp2)}  think_len={len(think_part2)}  pred={pred2}  {'✓' if ok2 else '✗'}")
    if after_think2:
        print(f"  answer_part: {after_think2[:200]}")

    # V2 without thinking (for direct-answer types)
    if t in ['numeral', 'gravity', 'unit_conv', 'cipher']:
        print(f"\n[V2 + thinking=False]")
        resp3, fin3 = call_api(v2_prompt, enable_thinking=False)
        pred3 = extract_boxed(resp3)
        ok3 = answers_match(pred3, gold)
        print(f"  finish={fin3}  len={len(resp3)}  pred={pred3}  {'✓' if ok3 else '✗'}")
        if resp3:
            print(f"  full_response: {resp3[:300]}")

    print(f"\n  SUMMARY: orig={'✓' if ok else '✗'}  v2_think={'✓' if ok2 else '✗'}" + 
          (f"  v2_nothink={'✓' if ok3 else '✗'}" if t in ['numeral','gravity','unit_conv','cipher'] else ""))
