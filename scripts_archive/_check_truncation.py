import pandas as pd

df = pd.read_csv('data/sft_full_cot.csv')

df['prompt_chars'] = df['prompt'].str.len()
df['think_chars'] = df['thinking'].str.len()
df['answer_chars'] = df['answer'].astype(str).str.len()

# Rough token estimate (chars / 3.5 + template overhead ~50 tokens)
df['est_tokens'] = (df['prompt_chars'] + df['think_chars'] + df['answer_chars']) / 3.5 + 50

print('=== Estimated total token lengths (with thinking) ===')
print(df['est_tokens'].describe())
print()

for thresh in [512, 1024, 2048]:
    n = (df['est_tokens'] > thresh).sum()
    print(f'Samples > {thresh} tokens: {n} / {len(df)} ({n/len(df)*100:.1f}%)')

print()

# E1 style (no thinking) - always fits
df['est_tokens_e1'] = (df['prompt_chars'] + df['answer_chars']) / 3.5 + 50
ne1 = (df['est_tokens_e1'] > 1024).sum()
print(f'E1 style (no thinking) > 1024: {ne1} / {len(df)}')
print()

# Thinking char length distribution
for thresh in [256, 512, 1024, 2048, 4096]:
    n = (df['think_chars'] > thresh).sum()
    print(f'Thinking > {thresh} chars: {n} ({n/len(df)*100:.1f}%)')

print()

# Truncation analysis for boxed-only loss at seq_len=1024
# </think> position = prompt_tokens + think_tokens + 1
df['prompt_tok_est'] = df['prompt_chars'] / 3.5 + 50
df['think_tok_est'] = df['think_chars'] / 3.5
df['think_close_pos'] = df['prompt_tok_est'] + df['think_tok_est'] + 1

# Case 1: </think> itself gets truncated -> loss=0 (wasted sample)
think_cutoff = (df['think_close_pos'] > 1024).sum()
print(f'Case 1 - </think> truncated (loss=0, wasted): {think_cutoff} ({think_cutoff/len(df)*100:.1f}%)')

# Case 2: </think> survives but boxed answer partially cut
partial = ((df['think_close_pos'] <= 1024) & (df['est_tokens'] > 1024)).sum()
print(f'Case 2 - </think> ok but boxed partially cut: {partial} ({partial/len(df)*100:.1f}%)')

# Case 3: Everything fits
fits = (df['est_tokens'] <= 1024).sum()
print(f'Case 3 - Fully fits in 1024: {fits} ({fits/len(df)*100:.1f}%)')

# How many loss tokens survive for truncated samples?
df['boxed_tokens_available'] = (1024 - df['think_close_pos']).clip(lower=0)
truncated = df[df['est_tokens'] > 1024]
if len(truncated) > 0:
    print(f'\nFor {len(truncated)} truncated samples:')
    print(f'  Average loss tokens available: {truncated["boxed_tokens_available"].mean():.1f}')
    print(f'  Samples with 0 loss tokens: {(truncated["boxed_tokens_available"] == 0).sum()}')
    print(f'  Samples with < 5 loss tokens: {(truncated["boxed_tokens_available"] < 5).sum()}')
