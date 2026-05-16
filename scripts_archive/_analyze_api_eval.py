#!/usr/bin/env python3
"""Detailed analysis of base model API eval results."""
import json
from collections import defaultdict

import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

results = []
with open('competition_data/base_model_eval.jsonl') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError:
            continue

print(f"Total results: {len(results)}")
print()

# Detailed per-type analysis
for t in sorted(set(r['type'] for r in results)):
    subset = [r for r in results if r['type'] == t]
    correct = sum(1 for r in subset if r['correct'])
    failures = [r for r in subset if not r['correct']]
    truncated = sum(1 for r in failures if r.get('finish_reason') == 'length')
    not_found = sum(1 for r in failures if r.get('predicted') == 'NOT_FOUND')
    
    print(f"=== {t}: {correct}/50 ({correct*2}%) ===")
    print(f"  Truncated: {truncated}/{len(failures)}, NOT_FOUND: {not_found}/{len(failures)}")
    
    # Wrong predictions (answered but wrong)
    wrong = [r for r in failures if r.get('predicted') != 'NOT_FOUND' and r.get('predicted') != 'ERROR']
    if wrong:
        print(f"  Wrong answer (has prediction but incorrect): {len(wrong)}")
        for r in wrong[:3]:
            print(f"    pred='{r['predicted'][:50]}' gold='{r['gold'][:50]}' finish={r['finish_reason']}")
    print()

# Summary table
print("=" * 60)
print("SUMMARY: Base Model (no LoRA) API Eval")
print("=" * 60)
print(f"{'Type':<12} {'Correct':>8} {'Rate':>7} {'Truncated':>10} {'Issue':>15}")
print("-" * 60)
total_c, total_n = 0, 0
for t in ['bit_ops', 'cipher', 'gravity', 'numeral', 'symbol', 'unit_conv']:
    subset = [r for r in results if r['type'] == t]
    if not subset:
        continue
    correct = sum(1 for r in subset if r['correct'])
    n = len(subset)
    trunc = sum(1 for r in subset if r.get('finish_reason') == 'length')
    total_c += correct
    total_n += n
    
    issue = ""
    if trunc > n * 0.5:
        issue = "TRUNCATION"
    elif correct < n * 0.2:
        issue = "HARD"
    
    print(f"{t:<12} {correct:>5}/{n:<2} {correct/n*100:>6.1f}% {trunc:>10} {issue:>15}")

print("-" * 60)
print(f"{'OVERALL':<12} {total_c:>5}/{total_n:<3} {total_c/total_n*100:>6.1f}%")

# Thinking length analysis
print()
print("=== Thinking/Content Length by Type ===")
for t in ['bit_ops', 'cipher', 'gravity', 'numeral', 'symbol', 'unit_conv']:
    subset = [r for r in results if r['type'] == t]
    think_lens = [r['thinking_len'] for r in subset if r.get('thinking_len', 0) > 0]
    content_lens = [r['content_len'] for r in subset if r.get('content_len', 0) > 0]
    if think_lens:
        avg_think = sum(think_lens) / len(think_lens)
        avg_content = sum(content_lens) / len(content_lens) if content_lens else 0
        print(f"  {t:<12} think_avg={avg_think:>7.0f} chars, content_avg={avg_content:>5.0f} chars")
