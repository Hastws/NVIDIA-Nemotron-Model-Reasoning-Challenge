#!/usr/bin/env python3
"""Investigate gravity recompute failures"""
import json, re

records = []
with open('data/cot_v2.jsonl') as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

gravity = [r for r in records if r['type'] == 'gravity']

# Check a specific failing case
for rec in gravity[:3]:
    print(f"=== ID {rec['id']} ===")
    print(f"PROMPT:\n{rec['prompt']}")
    print(f"\nTHINKING:\n{rec['thinking']}")
    print(f"\nANSWER: {rec['answer']}")
    
    # My regex extraction
    obs = re.findall(r'For t\s*=\s*([\d.]+)s?,\s*distance\s*=\s*([\d.]+)', rec['prompt'])
    qm = re.search(r'for t\s*=\s*([\d.]+)s', rec['prompt'], re.IGNORECASE)
    print(f"\nExtracted obs: {obs}")
    print(f"Query t: {qm.group(1) if qm else 'NOT FOUND'}")
    
    # Check - is the query t picking up one of the observation t values?
    all_t_matches = re.findall(r'for t\s*=\s*([\d.]+)s', rec['prompt'], re.IGNORECASE)
    print(f"All 'for t = XXs' matches: {all_t_matches}")
    
    # Get the actual query line
    lines = rec['prompt'].split('\n')
    for l in lines:
        if 'determine' in l.lower() or 'falling' in l.lower():
            print(f"Query line: {l}")
    print()

# Look at the thinking to see what g is computed
print("\n=== Verify thinking computation for first gravity record ===")
rec = gravity[0]
thinking = rec['thinking']
# Extract g values from thinking  
g_vals = re.findall(r'g\s*=\s*[\d.×/²]+\s*=\s*([\d.]+)', thinking)
print(f"g values in thinking: {g_vals}")
avg_match = re.search(r'Average g\s*=\s*([\d.]+)', thinking)
print(f"Average g in thinking: {avg_match.group(1) if avg_match else 'NOT FOUND'}")
result_match = re.search(r'd\s*=\s*[\d.×\s]+\s*=\s*([\d.]+)', thinking)
print(f"Final d computation: {result_match.group(1) if result_match else 'NOT FOUND'}")
