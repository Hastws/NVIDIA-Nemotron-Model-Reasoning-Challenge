"""Quick check: what's the total text length (prompt + thinking + answer) for programmatic CoT?"""
import csv, statistics

lengths = []
with open('/Users/hastws/work_space/NVIDIA Nemotron 模型推理挑战赛/data/sft_prog_with_cot.csv') as f:
    reader = csv.DictReader(f)
    for r in reader:
        # Simulate the full training text length (chars -> ~tokens/4)
        total = len(r['prompt']) + 50 + len(r.get('thinking','')) + 20 + len(r['answer']) + 10
        lengths.append(total)

print(f"Records: {len(lengths)}")
print(f"Char lengths: min={min(lengths)}, max={max(lengths)}, avg={statistics.mean(lengths):.0f}, median={statistics.median(lengths):.0f}")
print(f"Est tokens (chars/3.5): min={min(lengths)/3.5:.0f}, max={max(lengths)/3.5:.0f}, avg={statistics.mean(lengths)/3.5:.0f}")

# Check how many exceed 1024 tokens
over_1024 = sum(1 for l in lengths if l/3.5 > 1024)
over_512 = sum(1 for l in lengths if l/3.5 > 512)
print(f"\nOver 512 tokens: {over_512}/{len(lengths)} ({over_512/len(lengths)*100:.1f}%)")
print(f"Over 1024 tokens: {over_1024}/{len(lengths)} ({over_1024/len(lengths)*100:.1f}%)")
