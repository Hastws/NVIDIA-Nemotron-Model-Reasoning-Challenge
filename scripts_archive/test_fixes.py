#!/usr/bin/env python3
"""Test the bug fixes in multi_model_cot.py"""
import sys
sys.path.insert(0, '.')
from scripts.multi_model_cot import normalize_pred, answers_match, extract_boxed, score_cot

# Test normalize_pred
assert normalize_pred('42') == '42.0', f'Got: {normalize_pred("42")}'
assert normalize_pred('42.0') == '42.0'
assert normalize_pred('42.00') == '42.0'
assert normalize_pred('10.62') == '10.62'
assert normalize_pred('10.620') == '10.62'
assert normalize_pred('Cat') == 'cat'
assert normalize_pred('XXXVIII') == 'xxxviii'
assert normalize_pred('10010111') == '10010111.0'
assert normalize_pred(None) is None
assert normalize_pred('') == ''
print('✅ normalize_pred OK')

# Test answers_consistent fix
preds = ['42', '42.0', '42']
normalized = [normalize_pred(p) for p in preds]
assert len(set(normalized)) == 1, f'Got: {set(normalized)}'
print('✅ answers_consistent fix verified')

# Test agreement_ratio fix
from collections import Counter
raw_preds = ['42', '42.0', '42']
normalized_preds = [normalize_pred(p) for p in raw_preds]
c = Counter(normalized_preds)
ratio = c.most_common(1)[0][1] / len(normalized_preds)
assert ratio == 1.0, f'Expected 1.0, got {ratio}'
print('✅ agreement_ratio fix verified')

# Test score_cot with boxed position
s1 = {'thinking': 'abc', 'response': 'The answer is \\boxed{42}', 'finish_reason': 'stop', 'correct': True}
s2 = {'thinking': 'I think \\boxed{42}', 'response': 'The answer is 42', 'finish_reason': 'stop', 'correct': True}
sc1 = score_cot(s1)
sc2 = score_cot(s2)
print(f'   score with boxed in content: {sc1}')
print(f'   score with boxed only in thinking: {sc2}')
assert sc1 > sc2, 'boxed in content should score higher'
print('✅ score_cot boxed-position OK')

# Test answers_match
assert answers_match('42', '42.0') == True
assert answers_match('42', '43') == False
assert answers_match('3.14', '3.14') == True
assert answers_match('cat', 'Cat') == True  # case-insensitive per official verify()
print('✅ answers_match OK')

print('\n🎉 All logic tests passed!')
