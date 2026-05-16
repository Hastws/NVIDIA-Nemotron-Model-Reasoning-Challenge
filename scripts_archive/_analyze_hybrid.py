import pandas as pd

df = pd.read_csv('data/sft_cot_v2_hybrid.csv')
print('=== sft_cot_v2_hybrid.csv ===')
print(f'Total: {len(df)}')
print(f'Columns: {list(df.columns)}')
print(f'Type distribution:')
print(df['type'].value_counts().sort_index())

has_thinking = df['thinking'].notna() & (df['thinking'] != '')
print(f'\nWith thinking: {has_thinking.sum()}')
print(f'Answer-only: {(~has_thinking).sum()}')

thinking_lens = df.loc[has_thinking, 'thinking'].str.len()
print(f'\nThinking char lengths: min={thinking_lens.min()}, median={thinking_lens.median():.0f}, max={thinking_lens.max()}, mean={thinking_lens.mean():.0f}')

ans_lens = df['answer'].astype(str).str.len()
print(f'Answer char lengths: min={ans_lens.min()}, median={ans_lens.median():.0f}, max={ans_lens.max()}, mean={ans_lens.mean():.0f}')

print('\n=== Per-type thinking coverage ===')
for t in sorted(df['type'].unique()):
    sub = df[df['type']==t]
    has_t = sub['thinking'].notna() & (sub['thinking'] != '')
    t_lens = sub.loc[has_t.values, 'thinking'].str.len()
    avg_len = f', avg_chars={t_lens.mean():.0f}' if len(t_lens) > 0 else ''
    print(f'{t}: {has_t.sum()}/{len(sub)} ({has_t.sum()/len(sub)*100:.1f}%){avg_len}')

# Estimate token counts (rough: 1 token ~ 4 chars)
print('\n=== Estimated token lengths ===')
prompt_lens = df['prompt'].str.len() / 4
print(f'Prompt tokens (est): min={prompt_lens.min():.0f}, median={prompt_lens.median():.0f}, max={prompt_lens.max():.0f}')
thinking_toks = thinking_lens / 4
print(f'Thinking tokens (est): min={thinking_toks.min():.0f}, median={thinking_toks.median():.0f}, max={thinking_toks.max():.0f}')

# Full sequence length estimation for answer-only mode
# prompt + template overhead (~50 tokens) + answer
full_ao = prompt_lens + 50 + ans_lens / 4
print(f'\nFull seq (answer-only, est tokens): min={full_ao.min():.0f}, median={full_ao.median():.0f}, max={full_ao.max():.0f}, p95={full_ao.quantile(0.95):.0f}')

# Full sequence length with CoT
cot_rows = df[has_thinking]
full_cot = cot_rows['prompt'].str.len()/4 + 50 + cot_rows['thinking'].str.len()/4 + cot_rows['answer'].astype(str).str.len()/4
print(f'Full seq (with CoT, est tokens): min={full_cot.min():.0f}, median={full_cot.median():.0f}, max={full_cot.max():.0f}, p95={full_cot.quantile(0.95):.0f}')
