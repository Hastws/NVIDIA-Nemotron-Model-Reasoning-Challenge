"""
Analyze: how many questions does the base model likely FAIL 
because thinking consumes all tokens?

Based on T0 sampling data (9500 questions at max_tokens=3584).
And with the actual eval params: max_tokens=7680 from eval page.
"""

# From our T0 sampling data (conversation history)
# These are truncation rates at max_tokens=3584 (code default)
trunc_at_3584 = {
    "numeral":  0.003,   # 0.3%
    "gravity":  0.535,   # 53.5% of problems truncated
    "unit_conv": 0.799,  # 79.9%
    "cipher":   0.908,   # 90.8%  (from T0 data)
    "bit_ops":  0.980,   # 98.0%
    "symbol":   0.943,   # 94.3%
}

# From eval page: max_tokens=7680, max_model_len=8192
# But the evaluation page says max_tokens=7680 now
# The truncation rates at 7680 would be MUCH lower
# From previous analysis:
trunc_at_7680 = {
    "numeral":  0.0,     # essentially 0%
    "gravity":  0.199,   # 19.9% 
    "unit_conv": 0.404,  # 40.4%
    "cipher":   0.0,     # estimated (not measured directly at 7680 but much lower)
    "bit_ops":  0.937,   # 93.7% still truncated!
    "symbol":   0.869,   # 86.9% still truncated!
}

type_counts = {
    "numeral": 1576,
    "gravity": 1597,
    "unit_conv": 1594,
    "cipher": 1576,
    "bit_ops": 1602,
    "symbol": 1555,
}

print("=" * 70)
print("TRUNCATION IMPACT ANALYSIS")
print("=" * 70)

print("\n--- At max_tokens=7680 (eval page actual) ---")
total_trunc = 0
for t in sorted(type_counts):
    n = type_counts[t]
    rate = trunc_at_7680.get(t, 0)
    trunc_n = int(n * rate)
    total_trunc += trunc_n
    print(f"  {t:12s}: {n} questions, {rate*100:5.1f}% truncated = ~{trunc_n} lost")

print(f"\n  TOTAL LOST to truncation: ~{total_trunc}/9500 ({total_trunc*100/9500:.1f}%)")
print(f"  Maximum possible score if all non-truncated correct: {(9500-total_trunc)/9500:.3f}")

print("\n--- What if we could force ALL thinking to be SHORT (<500 tokens)? ---")
print("  Truncation → 0% for all types")
print(f"  Maximum possible: 1.000 (all 9500 correct)")
print(f"  Realistic upper bound (solver coverage): ")

solver_coverage = {
    "numeral":  1576/1576,
    "gravity":  1597/1597, 
    "unit_conv": 1594/1594,
    "cipher":   1551/1576,
    "bit_ops":  1423/1602,
    "symbol":   0/1555,
}

total_solvable = 0
for t in sorted(solver_coverage):
    n = type_counts[t]
    rate = solver_coverage[t]
    solvable = int(n * rate)
    total_solvable += solvable
    print(f"    {t:12s}: {solvable}/{n} ({rate*100:.1f}%)")
print(f"    TOTAL: {total_solvable}/9500 = {total_solvable/9500:.3f}")

print("\n--- Model base accuracy WITHOUT any LoRA (0.52) ---")
base_correct = int(0.52 * 9500)
print(f"  Base model gets ~{base_correct}/9500 correct")
print(f"  These are questions the model can ALREADY solve with long thinking")

print("\n" + "=" * 70)
print("KEY INSIGHT")
print("=" * 70)
print("""
The base model (0.52) already solves ~4940 questions WITH long thinking.
E1 LoRA (0.68) boosts this to ~6460.

The gain comes from the model learning WHAT to answer, not HOW to think.
But ~3000+ questions are lost to truncation (thinking too long, no boxed output).

If we could train the model to:
  1. Keep <think> section SHORT (or empty)
  2. ALWAYS output </think> before running out of tokens
  3. ALWAYS produce \\boxed{answer}

Then truncation losses → 0, and we capture all solvable questions.

APPROACHES:
  A) Train with empty thinking: <think></think>\\boxed{ans}
     → Model learns to skip thinking, go straight to answer
     → RISK: loses ability to reason through hard problems
     
  B) Train model to close thinking early:
     → Hard to control exact behavior
     
  C) Current E1/v29 approach: answer-only + enable_thinking
     → Template produces <think></think>\\boxed{ans}
     → This IS approach A! Model sees empty thinking in training
     → At inference, model may still generate long thinking (base behavior)
     
  D) PROBLEM: We can't control inference parameters
     → No thinking_budget in vLLM eval script
     → No system prompt to say "think briefly"
     → max_tokens=7680 is fixed
""")
