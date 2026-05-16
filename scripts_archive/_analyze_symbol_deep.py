import pandas as pd
import re

df = pd.read_csv('competition_data/train.csv')

# Better classification using regex
def classify(p):
    # symbol: contains special math symbols
    if re.search(r'[⊕⊗⊖△◇★◆●▲▽♠♣♦♥⊞⊟⊠]', p):
        return 'symbol'
    pl = p.lower()
    if 'cipher' in pl or 'encrypt' in pl or 'decrypt' in pl or 'encode' in pl or 'decode' in pl:
        return 'cipher'
    if 'gravit' in pl or 'planet' in pl:
        return 'gravity'
    if any(x in pl for x in ['base 2', 'base 8', 'base 10', 'base 16', 'base-2', 'base-8', 'base-10', 'base-16', 
                              'binary', 'octal', 'hexadecimal', 'decimal representation']):
        return 'numeral'
    if any(x in pl for x in ['meter', 'mile', 'gallon', 'liter', 'pound', 'kilogram', 'inch', 'foot', 'feet',
                              'yard', 'ounce', 'celsius', 'fahrenheit', 'kelvin', 'convert']):
        return 'unit_conv'
    if 'bit' in pl or 'xor' in pl or 'shift' in pl or 'bitwise' in pl:
        return 'bit_ops'
    return 'unknown'

df['type'] = df['prompt'].apply(classify)
print("=== Type Distribution ===")
print(df['type'].value_counts())
print()

# unknown IS symbol
symbol_df = df[df['type'] == 'unknown']
print(f"=== Symbol examples ({len(symbol_df)} total) ===\n")

# Show 15 diverse examples
for i, (idx, row) in enumerate(symbol_df.sample(min(15, len(symbol_df)), random_state=42).iterrows()):
    print(f"--- Example {i+1} ---")
    print(f"Prompt: {row['prompt'][:500]}")
    print(f"Answer: {row['answer']}")
    print()

# Analyze symbol patterns
print("=== Symbol pattern analysis ===")

# What symbols appear?
all_symbols = set()
for p in symbol_df['prompt']:
    found = re.findall(r'[⊕⊗⊖△◇★◆●▲▽♠♣♦♥⊞⊟⊠]', p)
    all_symbols.update(found)
print(f"Unique symbols found: {all_symbols}")

# Check if there are sub-patterns
# Pattern 1: symbol equations like "a ⊕ b = c"
eq_count = 0
table_count = 0
for p in symbol_df['prompt']:
    if 'table' in p.lower() or '|' in p:
        table_count += 1
    if '=' in p:
        eq_count += 1

print(f"Prompts with '=': {eq_count}/{len(symbol_df)}")
print(f"Prompts with table/|: {table_count}/{len(symbol_df)}")

# Answer distribution
print(f"\nAnswer types:")
print(f"  Numeric: {symbol_df['answer'].apply(lambda x: str(x).replace('.','').replace('-','').isdigit()).sum()}")
print(f"  Non-numeric: {(~symbol_df['answer'].apply(lambda x: str(x).replace('.','').replace('-','').isdigit())).sum()}")
print(f"\nSample answers: {symbol_df['answer'].head(20).tolist()}")
