#!/usr/bin/env python3
import csv, json, os

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

# 1. train_annotated.csv
with open(os.path.join(DATA, 'train_annotated.csv')) as f:
    rows = list(csv.DictReader(f))
matched = [r for r in rows if r['match'] == 'True']
sp = [r for r in matched if r.get('solution_process')]
print("=== 1. 规则解题数据: data/train_annotated.csv ===")
print(f"  总行数: {len(rows)}, 规则匹配: {len(matched)}, 有solution_process: {len(sp)}")
print(f"  字段: id, prompt, answer, type, solvable, solver_answer, solution_process, match, fail_reason")
if sp:
    print(f"  示例 solution_process: {sp[0]['solution_process'][:200]}")

# 2. train_dsl_rules.jsonl
print()
with open(os.path.join(DATA, 'train_dsl_rules.jsonl')) as f:
    dsl_rows = [json.loads(l) for l in f if l.strip()]
valid = [r for r in dsl_rows if r['score'] > 0]
print("=== 2. DSL/Compact规则: data/train_dsl_rules.jsonl ===")
print(f"  总行数: {len(dsl_rows)}, 有效(score>0): {len(valid)}")
print(f"  字段: id, type, dsl, score, reason, all_gens")
if valid:
    print(f"  示例 dsl: {valid[0]['dsl'][:200]}")

# 3. Full CoT (no boxed)
print()
test_file = os.path.join(DATA, 'cot_best4_nobox_test.jsonl')
full_file = os.path.join(DATA, 'cot_best4_nobox.jsonl')
if os.path.exists(test_file):
    with open(test_file) as f:
        test_rows = [json.loads(l) for l in f if l.strip()]
    correct = sum(1 for r in test_rows if r.get('correct'))
    print(f"=== 3. Full CoT (无boxed): data/cot_best4_nobox_test.jsonl (测试) ===")
    print(f"  行数: {len(test_rows)}, 正确: {correct}")
    print(f"  字段: id, type, answer, thinking, content, correct, tokens, n_candidates, best_score, ...")
    print(f"  示例 thinking: {test_rows[0]['thinking'][:200]}")

if os.path.exists(full_file):
    n = sum(1 for _ in open(full_file))
    print(f"\n=== 3b. Full CoT (全量生产中): data/cot_best4_nobox.jsonl ===")
    print(f"  已完成行数: {n} / 8121")
