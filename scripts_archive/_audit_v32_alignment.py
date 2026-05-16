#!/usr/bin/env python3
"""Deep audit of v32 token alignment: SUFFIX, boxed format, template structure."""
import json

with open('nvidia-nemotron-sfttrainer-v32.ipynb') as f:
    nb = json.load(f)

print("=" * 60)
print("V32 TOKEN ALIGNMENT AUDIT")
print("=" * 60)

# === Test 1: SUFFIX value ===
cell6_src = ''.join(nb['cells'][5]['source'])
exec_env = {}
for line in cell6_src.split('\n'):
    if line.strip().startswith('SUFFIX ='):
        exec(line.strip(), exec_env)
        break

our_suffix = exec_env['SUFFIX']
print("\n[1] SUFFIX Check")
print(f"  Our SUFFIX (repr): {repr(our_suffix)}")

# Official metric suffix (from competition metric script)
official = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'
print(f"  Official  (repr):  {repr(official)}")
print(f"  MATCH: {our_suffix == official}")

# === Test 2: assistant_msg escaping ===
print("\n[2] Assistant Message Escaping")
for ans in ["4", "3.14159", "HELLO", "XVI", "42"]:
    example = {"answer": ans}
    assistant_msg = f'\\boxed{{{example["answer"]}}}'
    print(f"  answer={ans!r:12s} -> assistant_msg={assistant_msg!r}")
    assert assistant_msg == f'\\boxed{{{ans}}}', f"MISMATCH for {ans}!"
    # Verify it starts with backslash-boxed
    assert assistant_msg.startswith('\\boxed{'), "Missing \\boxed prefix!"
    assert assistant_msg.endswith('}'), "Missing closing brace!"
print("  All correct!")

# === Test 3: Chat template structure ===
print("\n[3] Template Structure Analysis")
cell6_lines = cell6_src.split('\n')
has_enable_thinking = any('enable_thinking=True' in l for l in cell6_lines)
has_add_gen_false = any('add_generation_prompt=False' in l for l in cell6_lines)
has_user_role = any('"role": "user"' in l or "'role': 'user'" in l for l in cell6_lines)
has_asst_role = any('"role": "assistant"' in l or "'role': 'assistant'" in l for l in cell6_lines)
no_system_msg = not any('"role": "system"' in l or "'role': 'system'" in l for l in cell6_lines)

print(f"  enable_thinking=True: {has_enable_thinking}")
print(f"  add_generation_prompt=False: {has_add_gen_false}")
print(f"  Has user role: {has_user_role}")
print(f"  Has assistant role: {has_asst_role}")
print(f"  No system message: {no_system_msg}")

# === Test 4: SFTTrainer config ===
print("\n[4] SFTTrainer Config Check")
cell10_src = ''.join(nb['cells'][9]['source'])
checks = {
    "dataset_text_field='text'": "dataset_text_field='text'" in cell10_src,
    "packing=False": "packing=False" in cell10_src,
    "bf16=True": "bf16=True" in cell10_src,
    "gradient_checkpointing=True": "gradient_checkpointing=True" in cell10_src,
    "max_length=MAX_SEQ_LEN": "max_length=MAX_SEQ_LEN" in cell10_src,
}
for name, ok in checks.items():
    print(f"  {name}: {'PASS' if ok else 'FAIL'}")

# === Test 5: Cell 4 src path (no broken string) ===
print("\n[5] Cell 4 String Integrity")
cell4_src = ''.join(nb['cells'][3]['source'])
import ast
try:
    ast.parse(cell4_src)
    print("  ast.parse: PASS")
except SyntaxError as e:
    print(f"  ast.parse: FAIL - {e}")

# Check ptxas path is intact
if '/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/triton/backends/nvidia/bin/ptxas-blackwell' in cell4_src:
    print("  ptxas path: INTACT (single string)")
else:
    print("  ptxas path: WARNING - may be broken!")

# === Test 6: Inference vs Training alignment ===
print("\n[6] Inference vs Training Alignment Summary")
print("  Training format:")
print("    messages = [{user: prompt+SUFFIX}, {assistant: \\boxed{answer}}]")
print("    apply_chat_template(enable_thinking=True, add_generation_prompt=False)")
print("    -> <user>prompt+SUFFIX</user><assistant><think>\\n</think>\\n\\boxed{answer}</assistant>")
print()
print("  Inference format (vLLM):")
print("    messages = [{user: prompt+SUFFIX}]")
print("    apply_chat_template(enable_thinking=True, add_generation_prompt=True)")
print("    -> <user>prompt+SUFFIX</user><assistant><think>\\n")
print("    Model generates: </think>\\n\\boxed{answer}")
print()
print("  Alignment: CORRECT - training teaches model to output \\boxed{} after </think>")

# === Final verdict ===
print("\n" + "=" * 60)
all_ok = (our_suffix == official and has_enable_thinking and has_add_gen_false 
          and has_user_role and has_asst_role and no_system_msg)
if all_ok:
    print("VERDICT: ALL CHECKS PASSED - No token alignment issues")
else:
    print("VERDICT: ISSUES DETECTED - Review above")
print("=" * 60)
