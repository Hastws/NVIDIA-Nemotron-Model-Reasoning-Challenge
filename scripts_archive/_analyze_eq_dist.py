#!/usr/bin/env python3
"""Analyze operation distribution in eq_numeric and eq_symbolic training data."""

import csv
import re
from collections import Counter, defaultdict

DATA_PATH = 'data/sft_thinking.csv'

def analyze_eq_numeric(rows):
    """Analyze eq_numeric operation distribution from thinking CoT."""
    print(f"\n{'='*70}")
    print(f"  EQ_NUMERIC 操作分布分析 ({len(rows)} rows)")
    print(f"{'='*70}")

    # Extract operation from thinking
    op_counter = Counter()
    format_counter = Counter()
    reverse_counter = Counter()
    
    for row in rows:
        cot = row['thinking']
        if not cot:
            op_counter['[no thinking]'] += 1
            continue
        
        # Match "The operator 'X' maps to Y" pattern
        m = re.search(r"maps to (.+?)(?:\.|;|\n|$)", cot)
        if m:
            op_desc = m.group(1).strip().rstrip('.')
            op_counter[op_desc] += 1
            
            # Classify reverse
            if 'reverse' in op_desc.lower() or 'rev ' in op_desc.lower():
                reverse_counter['reverse'] += 1
            else:
                reverse_counter['plain'] += 1
            
            # Classify format
            if 'prefix' in op_desc.lower():
                if 'pos' in op_desc.lower():
                    format_counter['pos_prefix'] += 1
                else:
                    format_counter['prefix'] += 1
            elif 'suffix' in op_desc.lower():
                if 'pos' in op_desc.lower():
                    format_counter['pos_suffix'] += 1
                else:
                    format_counter['suffix'] += 1
            else:
                format_counter['none'] += 1
        else:
            # Try other patterns
            if 'concatenation' in cot.lower():
                if 'reverse' in cot.lower():
                    op_counter['reverse concatenation'] += 1
                else:
                    op_counter['concatenation'] += 1
                reverse_counter['plain'] += 1
                format_counter['none'] += 1
            else:
                op_counter['[unknown]'] += 1
    
    # Print operation distribution
    total = sum(op_counter.values())
    print(f"\n  --- 算子分布 (共 {total}) ---")
    
    # Group by base operation
    base_ops = defaultdict(int)
    for op, cnt in op_counter.items():
        # Normalize to base operation
        base = op.lower()
        # Remove format suffixes
        for fmt in ['; result prefixed with op symbol', '; result suffixed with op symbol',
                    '; positive results prefixed with op symbol', '; positive results suffixed with op symbol']:
            base = base.replace(fmt, '')
        base = base.strip()
        base_ops[base] += cnt
    
    for op, cnt in sorted(base_ops.items(), key=lambda x: -x[1]):
        bar = '█' * max(1, int(cnt / total * 80))
        print(f"  {op:50s} {cnt:4d} ({cnt/total*100:5.1f}%) {bar}")
    
    # Print reverse vs plain
    print(f"\n  --- 正向 vs 逆向 ---")
    for k in ['plain', 'reverse']:
        cnt = reverse_counter.get(k, 0)
        print(f"  {k:12s} {cnt:4d} ({cnt/total*100:5.1f}%)")
    
    # Print format distribution
    print(f"\n  --- 格式修饰符 ---")
    for k in ['none', 'prefix', 'suffix', 'pos_prefix', 'pos_suffix']:
        cnt = format_counter.get(k, 0)
        print(f"  {k:12s} {cnt:4d} ({cnt/total*100:5.1f}%)")


def analyze_eq_symbolic(rows):
    """Analyze eq_symbolic operation distribution from thinking CoT."""
    print(f"\n{'='*70}")
    print(f"  EQ_SYMBOLIC 操作分布分析 ({len(rows)} rows)")
    print(f"{'='*70}")

    op_counter = Counter()
    base_counter = Counter()
    format_counter = Counter()
    solver_layer = Counter()
    
    for row in rows:
        cot = row['thinking']
        if not cot:
            op_counter['[no thinking]'] += 1
            continue
        
        # Detect solver layer
        if 'Base-' in cot or 'base-' in cot:
            solver_layer['base-N'] += 1
            
            # Extract base
            m = re.search(r'[Bb]ase-(\d+)', cot)
            if m:
                base_counter[f"base-{m.group(1)}"] += 1
            
            # Extract operation within base-N
            # Pattern: "Operation: X" or "operation is X"
            m = re.search(r'(?:Operation|operation|Op)[:\s]+(.+?)(?:\.|;|\n|$)', cot)
            if m:
                op = m.group(1).strip().rstrip('.')
                op_counter[op] += 1
            else:
                # Try: "computed as X"
                m = re.search(r'computed as (\w[\w\s\-+×*]+?)(?:\.|;|\n|$)', cot)
                if m:
                    op_counter[m.group(1).strip()] += 1
                else:
                    # Extract from "Y = a OP b" patterns
                    ops_found = set()
                    for pattern, name in [
                        (r'(?:a|x)\s*[+]\s*(?:b|y)', 'addition'),
                        (r'(?:a|x)\s*[-−]\s*(?:b|y)', 'subtraction'),
                        (r'(?:a|x)\s*[×*]\s*(?:b|y)', 'multiplication'),
                        (r'\|(?:a|x)\s*[-−]\s*(?:b|y)\|', 'absolute difference'),
                        (r'concat', 'concatenation'),
                    ]:
                        if re.search(pattern, cot, re.I):
                            ops_found.add(name)
                    
                    if ops_found:
                        for o in ops_found:
                            op_counter[o] += 1
                    else:
                        op_counter['[base-N unknown op]'] += 1
            
            # Extract format
            if 'neg_prefix' in cot or 'negative prefix' in cot.lower():
                format_counter['neg_prefix'] += 1
            elif 'pos_prefix' in cot or 'positive prefix' in cot.lower():
                format_counter['pos_prefix'] += 1
            elif 'prefix' in cot.lower() and 'prepend' in cot.lower():
                format_counter['prefix'] += 1
            else:
                format_counter['none'] += 1
                
        elif 'charwise' in cot.lower() or 'char-wise' in cot.lower():
            solver_layer['charwise'] += 1
            m = re.search(r'charwise (\w+)', cot, re.I)
            if m:
                op_counter[f"charwise {m.group(1)}"] += 1
            else:
                op_counter['[charwise unknown]'] += 1
        elif 'concatenation' in cot.lower():
            solver_layer['string'] += 1
            if 'reverse' in cot.lower():
                op_counter['reverse concatenation'] += 1
            else:
                op_counter['string concatenation'] += 1
        else:
            solver_layer['unknown'] += 1
            op_counter['[unknown solver]'] += 1
    
    total = sum(op_counter.values())
    print(f"\n  --- Solver 层级 ---")
    for k, cnt in solver_layer.most_common():
        print(f"  {k:15s} {cnt:4d} ({cnt/len(rows)*100:5.1f}%)")
    
    print(f"\n  --- 算子分布 (共 {total}) ---")
    for op, cnt in op_counter.most_common():
        bar = '█' * max(1, int(cnt / total * 80))
        print(f"  {op:45s} {cnt:4d} ({cnt/total*100:5.1f}%) {bar}")
    
    print(f"\n  --- Base 分布 ---")
    for base, cnt in base_counter.most_common():
        bar = '█' * max(1, int(cnt / len(rows) * 60))
        print(f"  {base:10s} {cnt:4d} ({cnt/len(rows)*100:5.1f}%) {bar}")
    
    print(f"\n  --- 格式修饰符 ---")
    for k, cnt in format_counter.most_common():
        print(f"  {k:15s} {cnt:4d} ({cnt/len(rows)*100:5.1f}%)")


def deep_analyze_thinking(rows, type_name):
    """More detailed analysis by parsing the actual thinking text structure."""
    print(f"\n{'='*70}")
    print(f"  {type_name} — 详细 Thinking CoT 特征分析")
    print(f"{'='*70}")
    
    # Collect all unique "operation signature" lines
    op_lines = []
    for row in rows:
        cot = row['thinking']
        if not cot:
            continue
        for line in cot.split('\n'):
            line = line.strip()
            # Lines that describe the operation
            if any(kw in line.lower() for kw in ['maps to', 'operation:', 'rule:', 'computed as',
                                                    'base-', 'charwise', 'the operator']):
                op_lines.append(line)
    
    # Count unique patterns
    pattern_counter = Counter(op_lines)
    print(f"\n  Top 30 operation description lines ({len(pattern_counter)} unique):")
    for line, cnt in pattern_counter.most_common(30):
        print(f"  [{cnt:3d}] {line[:100]}")


# ═══════════════════════════════════════════════════════════════════════

def main():
    with open(DATA_PATH, encoding='utf-8') as f:
        all_rows = list(csv.DictReader(f))
    
    eq_num = [r for r in all_rows if r['type'] == 'eq_numeric']
    eq_sym = [r for r in all_rows if r['type'] == 'eq_symbolic']
    
    print(f"Total rows: {len(all_rows)}")
    print(f"eq_numeric: {len(eq_num)}")
    print(f"eq_symbolic: {len(eq_sym)}")
    
    analyze_eq_numeric(eq_num)
    analyze_eq_symbolic(eq_sym)
    
    # Detailed line-level analysis
    deep_analyze_thinking(eq_num, 'EQ_NUMERIC')
    deep_analyze_thinking(eq_sym, 'EQ_SYMBOLIC')

if __name__ == '__main__':
    main()
