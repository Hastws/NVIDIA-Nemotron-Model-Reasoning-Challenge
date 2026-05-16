#!/usr/bin/env python3
"""Analyze the Nemotron chat template to understand enable_thinking behavior."""

def simulate_template(content, has_reasoning_content=False, reasoning_content=""):
    """Simulate the Nemotron chat template for assistant messages."""
    if has_reasoning_content and reasoning_content.strip():
        content = "<think>\n" + reasoning_content + "\n</think>\n" + content
    else:
        if isinstance(content, str):
            if '<think>' not in content and '</think>' not in content:
                content = "<think></think>" + content
    return "<|im_start|>assistant\n" + content.strip() + "<|im_end|>"

# Test 1: V2 format — answer-only
print("=== TEST 1: V2 format (answer-only) ===")
result = simulate_template("\\boxed{4}")
print(repr(result))
print(result)
print()

# Test 2: CoT in content — manual <think> blocks
print("=== TEST 2: Manual <think> in content ===")
result = simulate_template("<think>\n2+2=4\n</think>\n\\boxed{4}")
print(repr(result))
print(result)
print()

# Test 3: Using reasoning_content field
print("=== TEST 3: reasoning_content field ===")
result = simulate_template("\\boxed{4}", True, "2+2=4")
print(repr(result))
print(result)
print()

# Test 4: generation prompt (inference time)
print("=== TEST 4: Inference prompts ===")
print("enable_thinking=True:  <|im_start|>assistant\\n<think>\\n")
print("enable_thinking=False: <|im_start|>assistant\\n<think></think>")
print()

# KEY ANALYSIS
print("=" * 60)
print("ANALYSIS")
print("=" * 60)
print()
print("1. Double-<think> hypothesis: DISPROVEN!")
print("   Template checks '<think>' in content before auto-insert.")
print("   If content has <think>CoT</think>, no auto-insert happens.")
print()
print("2. V2 training format (answer-only + enable_thinking=True):")
print("   -> <think></think>\\boxed{4}")
print("   NOTE: COMPACT empty think (no newlines)!")
print()
print("3. Manual CoT training format:")
print("   -> <think>\\n{cot}\\n</think>\\n\\boxed{4}")
print()
print("4. reasoning_content CoT format:")
print("   -> <think>\\n{cot}\\n</think>\\n\\boxed{4}")
print("   IDENTICAL to manual!")
print()
print("5. CRITICAL FORMAT GAP:")
print("   Training (V2): learns <think></think>\\boxed{answer}")
print("   Inference think=True: starts <think>\\n → model must generate </think>\\n\\boxed{}")  
print("   Inference think=False: starts <think></think> → model directly outputs \\boxed{}")
print()
print("   This explains WHY thinking=False always scores 0.66:")
print("   The inference format <think></think> EXACTLY matches training format!")
print()
print("6. FOR CoT TO WORK:")
print("   Training must match the inference format.")
print("   Inference with thinking=True starts: <think>\\n")
print("   So CoT training text should be: <think>\\n{cot}\\n</think>\\n\\boxed{4}")
print("   This IS what manual ChatML and reasoning_content produce.")
print("   So the format is CORRECT. The CoT failure must be from other causes.")
print()
print("7. BUT WAIT - the METRIC SCRIPT inference format:")
print("   The metric uses enable_thinking=True + max_tokens=3584")  
print("   So inference starts: <|im_start|>assistant\\n<think>\\n")
print("   Model generates: {thinking}\\n</think>\\n\\boxed{answer}")
print()
print("   If trained with V2 (answer-only):")
print("   Training: <think></think>\\boxed{4} (model never sees thinking content)")
print("   Inference: <think>\\n{base model generates some thinking}\\n</think>\\n\\boxed{}")
print("   The base model's native thinking ability handles the <think> block")
print()
print("   If trained with CoT:")
print("   Training: <think>\\n{our cot}\\n</think>\\n\\boxed{4}")
print("   This SHOULD teach the model better thinking... unless our CoT is BAD")
print("   Or unless 600 samples is too few to learn new reasoning patterns")
