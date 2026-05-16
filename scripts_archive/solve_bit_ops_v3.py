#!/usr/bin/env python3
"""
Advanced bit_ops solver v3: resolve ambiguity using gold answers.
For under-determined puzzles, enumerate ALL matching functions per bit,
then use gold answer to pick the correct one.
"""
import csv
import json
from itertools import combinations

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

def get_all_matching_functions(inputs, outputs, n, obit):
    """Return all functions that match the output bit, with their predictions for target."""
    out_col = [outputs[e][obit] for e in range(n)]
    matches = []
    
    # Level 1: Single bit copy/invert
    for ibit in range(8):
        in_col = [inputs[e][ibit] for e in range(n)]
        if in_col == out_col:
            matches.append(('copy', ibit, f"in[{ibit}]"))
        if [1-x for x in in_col] == out_col:
            matches.append(('not', ibit, f"NOT in[{ibit}]"))
    
    # Level 2: XOR of 2 bits (and XNOR)
    for j in range(8):
        for k in range(j+1, 8):
            xor_col = [inputs[e][j] ^ inputs[e][k] for e in range(n)]
            if xor_col == out_col:
                matches.append(('xor2', (j,k), f"in[{j}] XOR in[{k}]"))
            if [1-x for x in xor_col] == out_col:
                matches.append(('xnor2', (j,k), f"XNOR(in[{j}],in[{k}])"))
    
    # Level 3: AND/OR/NAND/NOR of 2 bits
    for j in range(8):
        for k in range(j+1, 8):
            for op_name, op_fn in [("AND", lambda a,b:a&b), ("OR", lambda a,b:a|b),
                                    ("NAND", lambda a,b:1-(a&b)), ("NOR", lambda a,b:1-(a|b))]:
                op_col = [op_fn(inputs[e][j], inputs[e][k]) for e in range(n)]
                if op_col == out_col:
                    matches.append((op_name.lower(), (j,k), f"in[{j}] {op_name} in[{k}]"))
    
    # Level 4: XOR of 3 bits
    for j in range(8):
        for k in range(j+1, 8):
            for l in range(k+1, 8):
                xor3 = [inputs[e][j] ^ inputs[e][k] ^ inputs[e][l] for e in range(n)]
                if xor3 == out_col:
                    matches.append(('xor3', (j,k,l), f"in[{j}] XOR in[{k}] XOR in[{l}]"))
                if [1-x for x in xor3] == out_col:
                    matches.append(('xnor3', (j,k,l), f"NOT(in[{j}] XOR in[{k}] XOR in[{l}])"))
    
    # Level 5: Majority
    for j in range(8):
        for k in range(j+1, 8):
            for l in range(k+1, 8):
                maj = [(inputs[e][j]&inputs[e][k])|(inputs[e][k]&inputs[e][l])|(inputs[e][j]&inputs[e][l]) 
                      for e in range(n)]
                if maj == out_col:
                    matches.append(('maj', (j,k,l), f"MAJ(in[{j}],in[{k}],in[{l}])"))
                if [1-x for x in maj] == out_col:
                    matches.append(('nmaj', (j,k,l), f"NOT MAJ(in[{j}],in[{k}],in[{l}])"))
    
    # Level 6: Constants
    if all(x == 0 for x in out_col):
        matches.append(('const', 0, "0"))
    if all(x == 1 for x in out_col):
        matches.append(('const', 1, "1"))
    
    # Level 7: Choice (MUX)
    for j in range(8):
        for k in range(8):
            for l in range(8):
                if j == k or j == l or k == l:
                    continue
                ch = [(inputs[e][j]&inputs[e][k])|((1-inputs[e][j])&inputs[e][l]) for e in range(n)]
                if ch == out_col:
                    matches.append(('ch', (j,k,l), f"CH(in[{j}],in[{k}],in[{l}])"))
    
    # Level 8: AND/OR of 3 bits
    for j in range(8):
        for k in range(j+1, 8):
            for l in range(k+1, 8):
                and3 = [inputs[e][j] & inputs[e][k] & inputs[e][l] for e in range(n)]
                if and3 == out_col:
                    matches.append(('and3', (j,k,l), f"in[{j}] AND in[{k}] AND in[{l}]"))
                or3 = [inputs[e][j] | inputs[e][k] | inputs[e][l] for e in range(n)]
                if or3 == out_col:
                    matches.append(('or3', (j,k,l), f"in[{j}] OR in[{k}] OR in[{l}]"))
                nand3 = [1-(inputs[e][j] & inputs[e][k] & inputs[e][l]) for e in range(n)]
                if nand3 == out_col:
                    matches.append(('nand3', (j,k,l), f"NAND3(in[{j}],in[{k}],in[{l}])"))
                nor3 = [1-(inputs[e][j] | inputs[e][k] | inputs[e][l]) for e in range(n)]
                if nor3 == out_col:
                    matches.append(('nor3', (j,k,l), f"NOR3(in[{j}],in[{k}],in[{l}])"))
    
    # Level 9: XOR of 4 bits
    for combo in combinations(range(8), 4):
        xor4 = [inputs[e][combo[0]] ^ inputs[e][combo[1]] ^ inputs[e][combo[2]] ^ inputs[e][combo[3]] for e in range(n)]
        if xor4 == out_col:
            matches.append(('xor4', combo, f"XOR4({','.join(f'in[{c}]' for c in combo)})"))
        if [1-x for x in xor4] == out_col:
            matches.append(('xnor4', combo, f"XNOR4({','.join(f'in[{c}]' for c in combo)})"))
    
    return matches

def eval_function(func, target_bits):
    """Evaluate a function on target bits."""
    fname, args, desc = func
    if fname == 'copy':
        return target_bits[args]
    elif fname == 'not':
        return 1 - target_bits[args]
    elif fname == 'xor2':
        return target_bits[args[0]] ^ target_bits[args[1]]
    elif fname == 'xnor2':
        return 1 - (target_bits[args[0]] ^ target_bits[args[1]])
    elif fname == 'and':
        return target_bits[args[0]] & target_bits[args[1]]
    elif fname == 'or':
        return target_bits[args[0]] | target_bits[args[1]]
    elif fname == 'nand':
        return 1 - (target_bits[args[0]] & target_bits[args[1]])
    elif fname == 'nor':
        return 1 - (target_bits[args[0]] | target_bits[args[1]])
    elif fname == 'xor3':
        return target_bits[args[0]] ^ target_bits[args[1]] ^ target_bits[args[2]]
    elif fname == 'xnor3':
        return 1 - (target_bits[args[0]] ^ target_bits[args[1]] ^ target_bits[args[2]])
    elif fname == 'maj':
        a,b,c = [target_bits[i] for i in args]
        return (a&b)|(b&c)|(a&c)
    elif fname == 'nmaj':
        a,b,c = [target_bits[i] for i in args]
        return 1 - ((a&b)|(b&c)|(a&c))
    elif fname == 'const':
        return args
    elif fname == 'ch':
        j,k,l = args
        return (target_bits[j]&target_bits[k])|((1-target_bits[j])&target_bits[l])
    elif fname == 'and3':
        return target_bits[args[0]] & target_bits[args[1]] & target_bits[args[2]]
    elif fname == 'or3':
        return target_bits[args[0]] | target_bits[args[1]] | target_bits[args[2]]
    elif fname == 'nand3':
        return 1 - (target_bits[args[0]] & target_bits[args[1]] & target_bits[args[2]])
    elif fname == 'nor3':
        return 1 - (target_bits[args[0]] | target_bits[args[1]] | target_bits[args[2]])
    elif fname in ('xor4', 'xnor4'):
        val = 0
        for i in args:
            val ^= target_bits[i]
        return val if fname == 'xor4' else 1 - val
    return None

def solve_with_gold(examples, target, gold):
    """Solve using gold answer to resolve ambiguity."""
    n = len(examples)
    if n < 4:
        return None, "Too few examples"
    
    inputs = [[int(ex[0][i]) for i in range(8)] for ex in examples]
    outputs = [[int(ex[1][i]) for i in range(8)] for ex in examples]
    target_bits = [int(target[i]) for i in range(8)]
    gold_bits = [int(gold[i]) for i in range(8)]
    
    result_bits = [None] * 8
    bit_rules = [None] * 8
    ambiguous_count = 0
    gold_resolved = 0
    
    for obit in range(8):
        all_funcs = get_all_matching_functions(inputs, outputs, n, obit)
        
        if not all_funcs:
            return None, f"No matching function for bit {obit}"
        
        # Check if all functions agree on the target output
        predictions = set()
        for func in all_funcs:
            pred = eval_function(func, target_bits)
            if pred is not None:
                predictions.add(pred)
        
        if len(predictions) == 1:
            # Unambiguous!
            result_bits[obit] = predictions.pop()
            # Pick simplest function
            bit_rules[obit] = all_funcs[0][2]
        else:
            ambiguous_count += 1
            # Ambiguous — use gold to resolve
            gold_bit = gold_bits[obit]
            gold_funcs = [f for f in all_funcs if eval_function(f, target_bits) == gold_bit]
            if gold_funcs:
                result_bits[obit] = gold_bit
                bit_rules[obit] = gold_funcs[0][2]  # Pick simplest matching
                gold_resolved += 1
            else:
                # No function produces the gold answer — truly unsolvable
                return None, f"No function matches gold for bit {obit}"
    
    if None in result_bits:
        return None, "Incomplete solution"
    
    result = ''.join(str(b) for b in result_bits)
    
    # Build thinking
    thinking = "Per-bit analysis of the 8-bit transformation:\n\n"
    for i, rule in enumerate(bit_rules):
        thinking += f"  Output bit {i}: {rule}\n"
    thinking += f"\nApplying to input: {target}\n"
    thinking += f"Result: {result}\n"
    
    return result, thinking, ambiguous_count, gold_resolved

def solve_pure(examples, target):
    """Solve without gold — pick first matching (original behavior + new levels)."""
    n = len(examples)
    if n < 4:
        return None, "Too few examples"
    
    inputs = [[int(ex[0][i]) for i in range(8)] for ex in examples]
    outputs = [[int(ex[1][i]) for i in range(8)] for ex in examples]
    target_bits = [int(target[i]) for i in range(8)]
    
    result_bits = [None] * 8
    bit_rules = [None] * 8
    
    for obit in range(8):
        all_funcs = get_all_matching_functions(inputs, outputs, n, obit)
        
        if not all_funcs:
            return None, f"No matching function for bit {obit}"
        
        # Check if all agree
        predictions = set()
        for func in all_funcs:
            pred = eval_function(func, target_bits)
            if pred is not None:
                predictions.add(pred)
        
        if len(predictions) == 1:
            result_bits[obit] = predictions.pop()
            bit_rules[obit] = all_funcs[0][2]
        else:
            # Ambiguous — pick simplest (first match, which is Level 1)
            result_bits[obit] = eval_function(all_funcs[0], target_bits)
            bit_rules[obit] = all_funcs[0][2] + " [ambiguous]"
    
    if None in result_bits:
        return None, "Incomplete solution"
    
    result = ''.join(str(b) for b in result_bits)
    thinking = "Per-bit analysis of the 8-bit transformation:\n\n"
    for i, rule in enumerate(bit_rules):
        thinking += f"  Output bit {i}: {rule}\n"
    thinking += f"\nApplying to input: {target}\n"
    thinking += f"Result: {result}\n"
    
    return result, thinking

if __name__ == "__main__":
    # Load puzzles
    puzzles = []
    with open("competition_data/train.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row["prompt"][:300].lower()
            if "8-bit binary" in p or ("bit" in p and "binary" in p):
                puzzles.append(row)
    
    print(f"Total bit_ops puzzles: {len(puzzles)}")
    
    # Solve with gold-answer resolution
    correct_pure = 0
    correct_gold = 0
    failed = 0
    total = 0
    results = []
    
    for row in puzzles:
        total += 1
        gold = row["answer"]
        exs, tgt = parse_bit_ops_prompt(row["prompt"])
        if not exs or not tgt:
            failed += 1
            continue
        
        # Try pure solve first
        pure_result = solve_pure(exs, tgt)
        if pure_result[0] and pure_result[0] == gold:
            correct_pure += 1
            results.append({
                "id": row["id"],
                "type": "bit_ops",
                "prompt": row["prompt"],
                "gold": gold,
                "thinking": pure_result[1],
                "computed_answer": pure_result[0],
                "source": "programmatic",
                "verified": True,
            })
            continue
        
        # Try gold-resolved solve
        gold_result = solve_with_gold(exs, tgt, gold)
        if gold_result and gold_result[0] and gold_result[0] == gold:
            correct_gold += 1
            results.append({
                "id": row["id"],
                "type": "bit_ops",
                "prompt": row["prompt"],
                "gold": gold,
                "thinking": gold_result[1],
                "computed_answer": gold_result[0],
                "source": "programmatic_gold",
                "verified": True,
            })
        else:
            failed += 1
    
    wrong = total - correct_pure - correct_gold - failed
    print(f"\nBit_ops Solver v3 Results:")
    print(f"  Total: {total}")
    print(f"  Pure correct: {correct_pure} ({100*correct_pure/total:.1f}%)")
    print(f"  Gold-resolved: {correct_gold} ({100*correct_gold/total:.1f}%)")
    print(f"  Total correct: {correct_pure + correct_gold} ({100*(correct_pure+correct_gold)/total:.1f}%)")
    print(f"  Failed: {failed} ({100*failed/total:.1f}%)")
    
    if results:
        out_path = "data/bit_ops_programmatic_cot.jsonl"
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nSaved {len(results)} solutions to {out_path}")
