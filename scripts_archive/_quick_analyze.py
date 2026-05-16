#!/usr/bin/env python3
"""Analyze sft_merged_v1.csv dataset composition."""
import pandas as pd

df = pd.read_csv('data/sft_merged_v1.csv')
has_thinking = df['thinking'].notna() & (df['thinking'].str.strip() != '') & (df['thinking'].str.strip().str.lower() != 'nan')

print(f'=== sft_merged_v1.csv 数据集分析 ===')
print(f'总行数: {len(df)}')
print(f'有 thinking: {has_thinking.sum()}')
print(f'纯 answer-only: {(~has_thinking).sum()}')

# Thinking 长度
thinking_df = df[has_thinking]
short = thinking_df['thinking'].str.len() < 50
medium = (thinking_df['thinking'].str.len() >= 50) & (thinking_df['thinking'].str.len() < 500)
long_t = thinking_df['thinking'].str.len() >= 500

print(f'\nThinking 长度分布:')
print(f'  极短 (<50 chars, compact rules): {short.sum()}')
print(f'  中等 (50-500 chars): {medium.sum()}')
print(f'  长 (>500 chars, full CoT): {long_t.sum()}')
print(f'\nThinking 长度统计:')
lens = thinking_df['thinking'].str.len()
print(f'  min={lens.min()}, median={lens.median():.0f}, mean={lens.mean():.0f}, max={lens.max()}')
