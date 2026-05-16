"""Gap analysis calculation."""
# Absolute numbers
for score, label in [(0.52, "Base"), (0.68, "E1"), (0.81, "Target")]:
    print(f"{label} ({score}): {int(score*9500)}/9500 correct")

print()
e1_correct = int(0.68 * 9500)
target_correct = int(0.81 * 9500)
gap = target_correct - e1_correct
print(f"Need {gap} MORE correct answers to reach 0.81")
print(f"That's {gap/9500*100:.1f}% of all questions")
print(f"Or roughly {gap//6:.0f} per type if evenly distributed")
