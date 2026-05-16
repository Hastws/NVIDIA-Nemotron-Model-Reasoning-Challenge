#!/usr/bin/env python3
"""
Advanced bit_ops solver using per-bit analysis.
For each output bit, find which input bit(s) determine it.
Covers: permutations, inversions, XOR of bit pairs, and more.
"""
import csv
import json

def parse_bit_ops_prompt(prompt):
    lines = prompt.strip().split('\n')
    examples = []
    target = None
    for line in lines:
        line = line.strip()
        if ' -> ' in line:
            parts = line.split(' -> ')
            if len(parts) == 2:
                inp, out = parts[0].strip(), parts[1].strip()
                if len(inp) == 8 and len(out) == 8 and all(c in '01' for c in inp+out):
                    examples.append((inp, out))
        elif 'determine' in line.lower() and ':' in line:
            t = line.split(':')[-1].strip()
            if len(t) == 8 and all(c in '01' for c in t):
                target = t
    return examples, target

def solve_per_bit(examples, target):
    """For each output bit position, determine it as a function of input bits."""
    n = len(examples)
    if n < 4:
        return None, "Too few examples"
    
    inputs = [[int(ex[0][i]) for i in range(8)] for ex in examples]
    outputs = [[int(ex[1][i]) for i in range(8)] for ex in examples]
    target_bits = [int(target[i]) for i in range(8)]
    
    result_bits = [None] * 8
    bit_rules = [None] * 8
    
    for obit in range(8):
        out_col = [outputs[e][obit] for e in range(n)]
        found = False
        
        # Level 1: output_bit = input_bit[j] or NOT input_bit[j]
        for ibit in range(8):
            in_col = [inputs[e][ibit] for e in range(n)]
            # Direct copy
            if in_col == out_col:
                result_bits[obit] = target_bits[ibit]
                bit_rules[obit] = f"out[{obit}] = in[{ibit}]"
                found = True
                break
            # Inverted
            if [1-x for x in in_col] == out_col:
                result_bits[obit] = 1 - target_bits[ibit]
                bit_rules[obit] = f"out[{obit}] = NOT in[{ibit}]"
                found = True
                break
        
        if found:
            continue
        
        # Level 2: output_bit = input_bit[j] XOR input_bit[k]
        for j in range(8):
            for k in range(j+1, 8):
                xor_col = [inputs[e][j] ^ inputs[e][k] for e in range(n)]
                if xor_col == out_col:
                    result_bits[obit] = target_bits[j] ^ target_bits[k]
                    bit_rules[obit] = f"out[{obit}] = in[{j}] XOR in[{k}]"
                    found = True
                    break
                # NOT(XOR)
                if [1-x for x in xor_col] == out_col:
                    result_bits[obit] = 1 - (target_bits[j] ^ target_bits[k])
                    bit_rules[obit] = f"out[{obit}] = NOT(in[{j}] XOR in[{k}])"
                    found = True
                    break
            if found:
                break
        
        if found:
            continue
        
        # Level 3: output_bit = input_bit[j] AND input_bit[k] (or OR, NAND, NOR)
        for j in range(8):
            for k in range(j+1, 8):
                for op_name, op_fn in [
                    ("AND", lambda a,b: a&b),
                    ("OR", lambda a,b: a|b),
                    ("NAND", lambda a,b: 1-(a&b)),
                    ("NOR", lambda a,b: 1-(a|b)),
                ]:
                    op_col = [op_fn(inputs[e][j], inputs[e][k]) for e in range(n)]
                    if op_col == out_col:
                        result_bits[obit] = op_fn(target_bits[j], target_bits[k])
                        bit_rules[obit] = f"out[{obit}] = in[{j}] {op_name} in[{k}]"
                        found = True
                        break
                if found:
                    break
            if found:
                break
        
        if found:
            continue
        
        # Level 4: XOR of 3 bits
        for j in range(8):
            for k in range(j+1, 8):
                for l in range(k+1, 8):
                    xor3_col = [inputs[e][j] ^ inputs[e][k] ^ inputs[e][l] for e in range(n)]
                    if xor3_col == out_col:
                        result_bits[obit] = target_bits[j] ^ target_bits[k] ^ target_bits[l]
                        bit_rules[obit] = f"out[{obit}] = in[{j}] XOR in[{k}] XOR in[{l}]"
                        found = True
                        break
                    if [1-x for x in xor3_col] == out_col:
                        result_bits[obit] = 1 - (target_bits[j] ^ target_bits[k] ^ target_bits[l])
                        bit_rules[obit] = f"out[{obit}] = NOT(in[{j}] XOR in[{k}] XOR in[{l}])"
                        found = True
                        break
                if found:
                    break
            if found:
                break
        
        if found:
            continue
        
        # Level 5: Majority(a,b,c) = (a&b)|(b&c)|(a&c)
        for j in range(8):
            for k in range(j+1, 8):
                for l in range(k+1, 8):
                    maj_col = [(inputs[e][j]&inputs[e][k])|(inputs[e][k]&inputs[e][l])|(inputs[e][j]&inputs[e][l]) 
                              for e in range(n)]
                    if maj_col == out_col:
                        tb = (target_bits[j]&target_bits[k])|(target_bits[k]&target_bits[l])|(target_bits[j]&target_bits[l])
                        result_bits[obit] = tb
                        bit_rules[obit] = f"out[{obit}] = MAJ(in[{j}],in[{k}],in[{l}])"
                        found = True
                        break
                    if [1-x for x in maj_col] == out_col:
                        tb = 1-((target_bits[j]&target_bits[k])|(target_bits[k]&target_bits[l])|(target_bits[j]&target_bits[l]))
                        result_bits[obit] = tb
                        bit_rules[obit] = f"out[{obit}] = NOT MAJ(in[{j}],in[{k}],in[{l}])"
                        found = True
                        break
                if found:
                    break
            if found:
                break
        
        if found:
            continue
            
        # Level 6: Constant 0 or 1
        if all(x == 0 for x in out_col):
            result_bits[obit] = 0
            bit_rules[obit] = f"out[{obit}] = 0"
            continue
        if all(x == 1 for x in out_col):
            result_bits[obit] = 1
            bit_rules[obit] = f"out[{obit}] = 1"
            continue
        
        # Level 7: Choice(a,b,c) = (a&b)|(~a&c)  
        for j in range(8):
            for k in range(8):
                for l in range(8):
                    if j == k or j == l or k == l:
                        continue
                    ch_col = [(inputs[e][j]&inputs[e][k])|((1-inputs[e][j])&inputs[e][l]) for e in range(n)]
                    if ch_col == out_col:
                        tb = (target_bits[j]&target_bits[k])|((1-target_bits[j])&target_bits[l])
                        result_bits[obit] = tb
                        bit_rules[obit] = f"out[{obit}] = CH(in[{j}],in[{k}],in[{l}])"
                        found = True
                        break
                if found:
                    break
            if found:
                break
        
        if not found:
            return None, f"Cannot determine output bit {obit}"
    
    if None in result_bits:
        return None, "Incomplete solution"
    
    result = ''.join(str(b) for b in result_bits)
    thinking = "Per-bit analysis of the transformation:\n\n"
    for i, rule in enumerate(bit_rules):
        thinking += f"  {rule}\n"
    thinking += f"\nInput: {target}\n"
    thinking += f"Output: {result}\n"
    
    return result, thinking

def solve_bit_ops(prompt, gold=None):
    examples, target = parse_bit_ops_prompt(prompt)
    if not examples or not target:
        return None, "Parse failed"
    return solve_per_bit(examples, target)

if __name__ == "__main__":
    correct = 0
    total = 0
    failed = 0
    wrong = 0
    
    results = []
    fail_reasons = {}
    
    with open("competition_data/train.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row["prompt"][:300].lower()
            if not ("8-bit binary" in p or ("bit" in p and "binary" in p)):
                continue
            total += 1
            gold = row["answer"]
            
            answer, info = solve_bit_ops(row["prompt"])
            
            if answer is None:
                failed += 1
                fail_reasons[info] = fail_reasons.get(info, 0) + 1
                continue
            
            if answer == gold:
                correct += 1
                results.append({
                    "id": row["id"],
                    "type": "bit_ops",
                    "prompt": row["prompt"],
                    "gold": gold,
                    "thinking": info,
                    "computed_answer": answer,
                    "source": "programmatic",
                    "verified": True,
                })
            else:
                wrong += 1
    
    print(f"\nAdvanced Bit_ops Solver Results:")
    print(f"  Total: {total}")
    print(f"  Correct: {correct} ({100*correct/total:.1f}%)")
    print(f"  Wrong: {wrong} ({100*wrong/total:.1f}%)")
    print(f"  Failed: {failed} ({100*failed/total:.1f}%)")
    print(f"\nFail reasons:")
    for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")
    
    if results:
        out_path = "data/bit_ops_programmatic_cot.jsonl"
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nSaved {len(results)} solutions to {out_path}")
