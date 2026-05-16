#!/usr/bin/env python3
"""
Programmatic solver for cipher-type puzzles.
These are substitution ciphers: each letter maps to another letter.
The examples give us the mapping (encrypted -> plaintext).
"""
import csv
import json
import re
from collections import Counter

def parse_cipher_prompt(prompt):
    """Parse a cipher prompt to extract examples and target."""
    lines = prompt.strip().split('\n')
    
    examples = []
    target = None
    
    for line in lines:
        line = line.strip()
        if ' -> ' in line and 'encrypt' not in line.lower() and 'example' not in line.lower():
            parts = line.split(' -> ')
            if len(parts) == 2:
                encrypted = parts[0].strip()
                plaintext = parts[1].strip()
                if encrypted and plaintext:
                    examples.append((encrypted, plaintext))
        elif line.startswith('Now, decrypt') or line.startswith('Now decrypt'):
            # Extract target text after colon
            match = re.search(r':\s*(.+)', line)
            if match:
                target = match.group(1).strip()
    
    return examples, target

def build_mapping(examples):
    """Build char-to-char mapping from examples."""
    mapping = {}  # encrypted_char -> plaintext_char
    conflicts = {}
    
    for encrypted, plaintext in examples:
        enc_words = encrypted.split()
        plain_words = plaintext.split()
        
        if len(enc_words) != len(plain_words):
            continue
        
        for ew, pw in zip(enc_words, plain_words):
            if len(ew) != len(pw):
                continue
            for ec, pc in zip(ew, pw):
                if ec in mapping:
                    if mapping[ec] != pc:
                        conflicts[ec] = conflicts.get(ec, set())
                        conflicts[ec].add(mapping[ec])
                        conflicts[ec].add(pc)
                else:
                    mapping[ec] = pc
    
    # Space maps to space
    mapping[' '] = ' '
    
    return mapping, conflicts

def decrypt(text, mapping):
    """Decrypt text using the mapping."""
    result = []
    unmapped = []
    for c in text:
        if c in mapping:
            result.append(mapping[c])
        else:
            result.append(f'?{c}?')
            unmapped.append(c)
    return ''.join(result), unmapped

def generate_thinking(examples, mapping, target, decrypted):
    """Generate step-by-step CoT for the decryption."""
    lines = []
    lines.append("I need to decrypt the ciphertext using the substitution cipher pattern from the examples.")
    lines.append("")
    lines.append("Step 1: Build the substitution mapping from the examples.")
    lines.append("From the example pairs, I can map each encrypted letter to its plaintext equivalent:")
    
    # Show sorted mapping
    sorted_map = sorted(mapping.items(), key=lambda x: x[0])
    map_strs = [f"{k}->{v}" for k, v in sorted_map if k != ' ']
    lines.append("  " + ", ".join(map_strs))
    
    lines.append("")
    lines.append(f"Step 2: Apply the mapping to decrypt: '{target}'")
    
    words = target.split()
    for word in words:
        decrypted_word = ''.join(mapping.get(c, f'?{c}?') for c in word)
        char_maps = [f"{c}->{mapping.get(c, '?')}" for c in word]
        lines.append(f"  '{word}' -> {', '.join(char_maps)} -> '{decrypted_word}'")
    
    lines.append("")
    lines.append(f"Result: {decrypted}")
    
    return '\n'.join(lines)

def solve_cipher(prompt):
    """Solve a cipher puzzle and return (answer, thinking) or (None, error)."""
    examples, target = parse_cipher_prompt(prompt)
    
    if not examples:
        return None, "No examples found"
    if not target:
        return None, "No target found"
    
    mapping, conflicts = build_mapping(examples)
    
    if conflicts:
        # Try to resolve by majority vote
        for ec, pcs in conflicts.items():
            # Use the most common mapping
            all_mappings = []
            for encrypted, plaintext in examples:
                enc_words = encrypted.split()
                plain_words = plaintext.split()
                for ew, pw in zip(enc_words, plain_words):
                    if len(ew) != len(pw):
                        continue
                    for e, p in zip(ew, pw):
                        if e == ec:
                            all_mappings.append(p)
            if all_mappings:
                mapping[ec] = Counter(all_mappings).most_common(1)[0][0]
    
    decrypted, unmapped = decrypt(target, mapping)
    
    if unmapped:
        return None, f"Unmapped chars: {unmapped}"
    
    if '?' in decrypted:
        return None, f"Incomplete decryption: {decrypted}"
    
    thinking = generate_thinking(examples, mapping, target, decrypted)
    
    return decrypted, thinking

# --- Main: solve all cipher puzzles ---
if __name__ == "__main__":
    correct = 0
    total = 0
    failed = 0
    errors = Counter()
    
    results = []
    
    with open("competition_data/train.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = row["prompt"]
            p = prompt[:300].lower()
            if not ("encrypt" in p or "cipher" in p or "secret code" in p):
                continue
            
            total += 1
            answer, thinking = solve_cipher(prompt)
            
            if answer is None:
                failed += 1
                errors[thinking] += 1
                continue
            
            gold = row["answer"]
            if answer.strip().lower() == gold.strip().lower():
                correct += 1
                results.append({
                    "id": row["id"],
                    "type": "cipher",
                    "prompt": prompt,
                    "gold": gold,
                    "thinking": thinking,
                    "computed_answer": answer,
                    "source": "programmatic",
                    "verified": True,
                })
            else:
                errors[f"wrong_answer"] += 1
                if total <= 5 or (total % 200 == 0):
                    print(f"WRONG: gold='{gold}' vs computed='{answer}'")
                    print(f"  Prompt: {prompt[:200]}...")
    
    print(f"\nCipher Solver Results:")
    print(f"  Total: {total}")
    print(f"  Correct: {correct} ({100*correct/total:.1f}%)")
    print(f"  Failed: {failed}")
    print(f"  Wrong: {total - correct - failed}")
    print(f"  Errors: {dict(errors)}")
    
    # Save results
    if results:
        out_path = "data/cipher_programmatic_cot.jsonl"
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nSaved {len(results)} correct solutions to {out_path}")
