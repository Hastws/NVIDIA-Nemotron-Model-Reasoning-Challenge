"""
Test: Can the base Nemotron model solve problems when given our rule-based CoT as guidance?

Three test modes:
  A) "rules-in-prompt": Inject extracted rules into user prompt, ask model to apply them
  B) "one-shot-cot": Provide ONE worked example (different problem) as few-shot, then ask new problem
  C) "bare-baseline": No help at all — just the original problem (for comparison)

Usage:
  python scripts/test_cot_guided.py --types bit_ops cipher eq_numeric --n 5
"""
import os
import re
import csv
import sys
import json
import math
import time
import argparse
import random
from collections import defaultdict

from openai import OpenAI

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "nvidia/nemotron-3-nano-30b-a3b"

PROMPT_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def extract_answer(text):
    """Extract answer from \\boxed{...}"""
    if not text:
        return "NOT_FOUND"
    matches = re.findall(r'\\boxed\{([^}]*)(?:\}|$)', text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        return non_empty[-1] if non_empty else matches[-1].strip()
    return "NOT_FOUND"


_BINARY_RE = re.compile(r'^[01]+$')

def verify(gold, predicted):
    """Official-style verification."""
    gold = str(gold).strip()
    predicted = str(predicted).strip()
    if predicted.lower() == gold.lower():
        return True
    if len(gold) > 1 and _BINARY_RE.match(gold):
        return False
    try:
        return math.isclose(float(gold), float(predicted), rel_tol=1e-2, abs_tol=1e-5)
    except:
        return False


def load_train_data(csv_path):
    """Load training data with thinking."""
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def call_nemotron(client, messages, temperature=0.0, max_tokens=3584):
    """Call Nemotron API with enable_thinking."""
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1.0,
            extra_body={"thinking": {"type": "enabled"}},
        )
        msg = resp.choices[0].message
        thinking = getattr(msg, 'reasoning_content', None) or ""
        content = msg.content or ""
        return thinking, content
    except Exception as e:
        return "", f"ERROR: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST MODES
# ═══════════════════════════════════════════════════════════════════════════════

def test_bare_baseline(client, row):
    """Mode C: No help, just the original problem + suffix."""
    prompt = row['prompt'] + PROMPT_SUFFIX
    messages = [{"role": "user", "content": prompt}]
    return call_nemotron(client, messages)


def test_rules_in_prompt(client, row):
    """Mode A: Inject our extracted rules into the prompt.
    
    Format:
    [Original problem]
    
    --- Analysis ---
    Here are the rules I've extracted from the examples above:
    [Our CoT rules section]
    
    Now apply these rules to solve the problem.
    [PROMPT_SUFFIX]
    """
    thinking = row.get('thinking', '')
    if not thinking or thinking.strip() == '' or thinking.strip().lower() == 'nan':
        return None, None  # skip — no rules available
    
    # Extract just the "rules" part (before "For input" / "For t=" / execution section)
    rules_only = extract_rules_section(thinking, row['type'])
    
    prompt = (
        f"{row['prompt']}\n\n"
        f"--- Analysis ---\n"
        f"Here are the rules I've extracted from the examples above:\n"
        f"{rules_only}\n\n"
        f"Now apply these rules to determine the answer."
        f"{PROMPT_SUFFIX}"
    )
    messages = [{"role": "user", "content": prompt}]
    return call_nemotron(client, messages)


def test_full_cot_in_prompt(client, row):
    """Mode A2: Inject the FULL CoT (rules + execution) into the prompt."""
    thinking = row.get('thinking', '')
    if not thinking or thinking.strip() == '' or thinking.strip().lower() == 'nan':
        return None, None
    
    prompt = (
        f"{row['prompt']}\n\n"
        f"--- Step-by-step solution ---\n"
        f"{thinking.strip()}\n\n"
        f"Based on the above analysis, give the final answer."
        f"{PROMPT_SUFFIX}"
    )
    messages = [{"role": "user", "content": prompt}]
    return call_nemotron(client, messages)


def test_one_shot(client, row, example_row):
    """Mode B: One-shot with a different worked example."""
    thinking = example_row.get('thinking', '')
    if not thinking or thinking.strip() == '' or thinking.strip().lower() == 'nan':
        return None, None
    
    # Build few-shot: show example problem + its solution, then the target problem
    system = (
        "You are solving reasoning puzzles. I'll show you one worked example first, "
        "then ask you to solve a similar problem using the same approach."
    )
    
    few_shot_user = (
        f"Example problem:\n{example_row['prompt']}\n\n"
        f"Example solution:\n{thinking.strip()}\n\n"
        f"The answer is: \\boxed{{{example_row['answer']}}}"
    )
    
    target_user = (
        f"Now solve this similar problem using the same approach:\n\n"
        f"{row['prompt']}"
        f"{PROMPT_SUFFIX}"
    )
    
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": few_shot_user},
        {"role": "assistant", "content": f"I've worked through the example. Now please give me the next problem."},
        {"role": "user", "content": target_user},
    ]
    return call_nemotron(client, messages)


def extract_rules_section(thinking, problem_type):
    """Extract only the rules part, before the execution on target."""
    lines = thinking.strip().split('\n')
    rules_lines = []
    for line in lines:
        # Stop at execution markers
        lower = line.lower().strip()
        if lower.startswith('for input ') or lower.startswith('for t=') or lower.startswith('for t ='):
            break
        if lower.startswith('converting ') and problem_type == 'numeral':
            break
        if 'apply' in lower and 'rules' in lower:
            break
        # For eq_numeric/cipher, "Result:" is the conclusion
        if lower.startswith('result:'):
            break
        rules_lines.append(line)
    
    return '\n'.join(rules_lines).strip()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Test CoT-guided inference on Nemotron base model")
    parser.add_argument("--types", nargs="+", default=["bit_ops", "cipher", "eq_numeric"],
                        help="Problem types to test")
    parser.add_argument("--n", type=int, default=5, help="Number of problems per type")
    parser.add_argument("--modes", nargs="+", default=["bare", "rules", "full_cot"],
                        choices=["bare", "rules", "full_cot", "one_shot"],
                        help="Test modes: bare, rules, full_cot, one_shot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", default="data/sft_thinking.csv")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"{'='*70}")
    print(f"  CoT-Guided Inference Test")
    print(f"  Model: {MODEL}")
    print(f"  Types: {args.types}")
    print(f"  N per type: {args.n}")
    print(f"  Modes: {args.modes}")
    print(f"{'='*70}\n")

    # Load data
    all_rows = load_train_data(args.data)
    by_type = defaultdict(list)
    for row in all_rows:
        t = row.get('type', 'unknown')
        if t in args.types and row.get('thinking', '').strip() and row['thinking'].strip().lower() != 'nan':
            by_type[t].append(row)

    for t in args.types:
        print(f"  {t}: {len(by_type[t])} rows with thinking")

    # Select test samples
    test_samples = {}
    for t in args.types:
        pool = by_type[t]
        if len(pool) <= args.n:
            test_samples[t] = pool
        else:
            test_samples[t] = random.sample(pool, args.n)

    # Init API client
    client = OpenAI(base_url=BASE_URL, api_key=NVIDIA_API_KEY)

    # Run tests
    results = defaultdict(lambda: defaultdict(list))  # type -> mode -> [bool]
    
    for t in args.types:
        samples = test_samples[t]
        # Get a separate example for one-shot (first one not in test set, or first from pool)
        one_shot_example = None
        if "one_shot" in args.modes:
            for r in by_type[t]:
                if r not in samples:
                    one_shot_example = r
                    break
            if not one_shot_example and len(by_type[t]) > 0:
                one_shot_example = by_type[t][0]

        for i, row in enumerate(samples):
            print(f"\n{'━'*70}")
            print(f"  [{t}] Problem {i+1}/{len(samples)} (id={row['id']})")
            print(f"  Gold answer: {row['answer']}")
            print(f"{'━'*70}")
            
            # Show our CoT for reference
            thinking = row.get('thinking', '')
            print(f"\n  📋 Our CoT ({len(thinking)} chars):")
            for line in thinking.strip().split('\n')[:8]:
                print(f"    {line}")
            if thinking.count('\n') > 8:
                print(f"    ... ({thinking.count(chr(10))+1} lines total)")

            for mode in args.modes:
                print(f"\n  --- Mode: {mode} ---")
                t0 = time.time()

                if mode == "bare":
                    model_thinking, model_content = test_bare_baseline(client, row)
                elif mode == "rules":
                    model_thinking, model_content = test_rules_in_prompt(client, row)
                elif mode == "full_cot":
                    model_thinking, model_content = test_full_cot_in_prompt(client, row)
                elif mode == "one_shot":
                    if one_shot_example:
                        model_thinking, model_content = test_one_shot(client, row, one_shot_example)
                    else:
                        model_thinking, model_content = None, None
                
                elapsed = time.time() - t0

                if model_content is None:
                    print(f"    ⏭️  SKIPPED (no CoT data)")
                    continue

                predicted = extract_answer(model_content)
                correct = verify(row['answer'], predicted)
                status = "✅" if correct else "❌"
                results[t][mode].append(correct)

                print(f"    {status} predicted={predicted} | gold={row['answer']} | {elapsed:.1f}s")
                
                # Show model thinking (truncated)
                if model_thinking:
                    think_preview = model_thinking[:300].replace('\n', '\n      ')
                    print(f"    🧠 Thinking: {think_preview}")
                    if len(model_thinking) > 300:
                        print(f"      ... ({len(model_thinking)} chars)")
                
                # Show model content (truncated)
                content_preview = model_content[:200].replace('\n', '\n      ')
                print(f"    📝 Content: {content_preview}")
                
                time.sleep(0.5)  # rate limit

    # ═══════════════════════════════════════════════════════════════════════════
    #  SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*70}")
    print(f"  📊 SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Type':<15} {'Mode':<12} {'Correct':>8} {'Total':>6} {'Acc':>8}")
    print(f"  {'-'*51}")
    
    for t in args.types:
        for mode in args.modes:
            r = results[t][mode]
            if r:
                n_correct = sum(r)
                n_total = len(r)
                acc = n_correct / n_total * 100
                print(f"  {t:<15} {mode:<12} {n_correct:>5}/{n_total:<3} {acc:>6.1f}%")

    # Aggregate by mode
    print(f"\n  {'─'*51}")
    print(f"  {'AGGREGATE':<15} {'Mode':<12} {'Correct':>8} {'Total':>6} {'Acc':>8}")
    print(f"  {'─'*51}")
    for mode in args.modes:
        all_correct = sum(sum(results[t][mode]) for t in args.types)
        all_total = sum(len(results[t][mode]) for t in args.types)
        if all_total > 0:
            acc = all_correct / all_total * 100
            print(f"  {'ALL':<15} {mode:<12} {all_correct:>5}/{all_total:<3} {acc:>6.1f}%")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
