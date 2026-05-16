"""Fix Cell 7 (data loading) in the 2-stage SFT notebook."""
import json

NB_PATH = 'nvidia-nemotron-2stage-sft.ipynb'

NEW_SOURCE = r'''print("=== CELL V84 DATA LOADING ===")  # Version marker

MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
COMP_DATA = '/kaggle/input/nvidia-nemotron-3-reasoning-challenge'

# --- Exhaustive search for training data ---
import glob as _glob

TARGET_FILE = 'sft_merged_v1.csv'

# Method 1: Check known candidates
_COT_CANDIDATES = [
    '/kaggle/input/prog-cot-training-data',
    '/kaggle/input/datasets/hastws/prog-cot-training-data',
]
COT_DATA = None
for d in _COT_CANDIDATES:
    csv_path = os.path.join(d, TARGET_FILE)
    if os.path.isfile(csv_path):
        COT_DATA = d
        print(f"Found via candidate: {csv_path}")
        break

# Method 2: Glob search under /kaggle/input/
if COT_DATA is None:
    print("Candidate dirs failed. Searching /kaggle/input/ recursively...")
    matches = _glob.glob(f'/kaggle/input/**/{TARGET_FILE}', recursive=True)
    if matches:
        COT_DATA = os.path.dirname(matches[0])
        print(f"Found via glob: {matches[0]}")
    else:
        print(f"ERROR: {TARGET_FILE} not found anywhere under /kaggle/input/")
        print("\n--- /kaggle/input/ contents ---")
        for root, dirs, files in os.walk('/kaggle/input/'):
            level = root.replace('/kaggle/input/', '').count(os.sep)
            indent = ' ' * 2 * level
            print(f'{indent}{os.path.basename(root)}/')
            if level < 3:
                subindent = ' ' * 2 * (level + 1)
                for f in files[:20]:
                    print(f'{subindent}{f}')
        raise FileNotFoundError(f"{TARGET_FILE} not found under /kaggle/input/. See listing above.")

print(f"COT_DATA = {COT_DATA}")
print(f"  Files: {os.listdir(COT_DATA)[:10]}")

# Load the merged dataset
train_df = pl.read_csv(f'{COT_DATA}/{TARGET_FILE}')

print(f"{'='*60}")
print(f"  Loaded: {TARGET_FILE} — {len(train_df)} rows")
print(f"{'='*60}")

# Statistics
pdf = train_df.to_pandas()
has_thinking = pdf['thinking'].fillna('').str.strip().str.len() > 0
n_with = has_thinking.sum()
n_without = (~has_thinking).sum()
print(f"\n  With thinking: {n_with} ({n_with/len(pdf)*100:.1f}%)")
print(f"  Without thinking (answer-only): {n_without} ({n_without/len(pdf)*100:.1f}%)")

# Classify thinking types
short_mask = has_thinking & (pdf['thinking'].str.len() < 50)
long_mask = has_thinking & (pdf['thinking'].str.len() >= 50)
print(f"  - Compact rules (<50 chars): {short_mask.sum()}")
print(f"  - Full CoT (≥50 chars): {long_mask.sum()}")

# Show thinking length distribution
print(f"\n  Thinking length stats (non-empty):")
lengths = pdf.loc[has_thinking, 'thinking'].str.len()
print(f"    min={lengths.min()}, median={lengths.median():.0f}, mean={lengths.mean():.0f}, max={lengths.max()}")

# Check for any data issues
print(f"\n  --- Sanity checks ---")
print(f"  Empty prompt: {(pdf['prompt'].fillna('').str.len() == 0).sum()}")
print(f"  Empty answer: {(pdf['answer'].fillna('').astype(str).str.len() == 0).sum()}")
boxed_in_thinking = pdf.loc[has_thinking, 'thinking'].str.contains(r'\\boxed', regex=True, na=False).sum()
print(f"  \\boxed in thinking: {boxed_in_thinking}")
print(f"  Columns: {list(pdf.columns)}")

# =============================================
#  TYPE INFERENCE FUNCTION
# =============================================
def _infer_type(prompt):
    p = prompt.lower()
    if 'bit manipulation' in p or '8-bit binary' in p:
        return 'bit_ops'
    elif 'numeral system' in p:
        return 'numeral'
    elif 'encrypt' in p or 'decrypt' in p:
        return 'cipher'
    elif 'gravitational' in p or 'gravity' in p or 'free-fall' in p:
        return 'gravity'
    elif 'unit' in p and ('convert' in p or 'conversion' in p):
        return 'unit_conv'
    elif 'symbol' in p or 'transformation rule' in p:
        return 'symbol'
    return 'unknown'

# =============================================
#  HOLDOUT SPLIT (for per-type evaluation)
# =============================================
if HOLDOUT_ENABLED:
    import re as _re
    
    # Load original train.csv for ground truth + type inference
    _orig_train = pl.read_csv(f'{COMP_DATA}/train.csv').to_pandas()
    _orig_train['type'] = _orig_train['prompt'].apply(_infer_type)
    
    # Stratified holdout: HOLDOUT_N_PER_TYPE per type (ALL 6 types for diagnosis)
    _holdout_parts = []
    for _t in sorted(_orig_train['type'].unique()):
        _type_df = _orig_train[_orig_train['type'] == _t]
        _sampled = _type_df.sample(n=min(HOLDOUT_N_PER_TYPE, len(_type_df)), random_state=999)
        _holdout_parts.append(_sampled)
    holdout_df = pd.concat(_holdout_parts).reset_index(drop=True)
    holdout_ids = set(holdout_df['id'].tolist())
    
    print(f"\n{'='*60}")
    print(f"  HOLDOUT: {len(holdout_df)} samples ({HOLDOUT_N_PER_TYPE}/type)")
    print(f"{'='*60}")
    for _t in sorted(holdout_df['type'].unique()):
        print(f"    {_t}: {(holdout_df['type']==_t).sum()}")
    
    # Filter training data to exclude holdout
    _before = len(train_df)
    _keep_mask = ~train_df.to_pandas()['id'].isin(holdout_ids)
    train_df = pl.from_pandas(train_df.to_pandas()[_keep_mask.values].reset_index(drop=True))
    print(f"\n  Training data: {_before} -> {len(train_df)} (removed {_before - len(train_df)} holdout overlap)")
else:
    holdout_df = None
    print("\nHoldout evaluation: DISABLED")

# =============================================
#  TYPE FILTER (train on subset of types)
# =============================================
if TRAIN_TYPES:
    _train_pdf = train_df.to_pandas()
    _train_pdf['_type'] = _train_pdf['prompt'].apply(_infer_type)
    _before_type = len(_train_pdf)
    _train_pdf = _train_pdf[_train_pdf['_type'].isin(TRAIN_TYPES)].reset_index(drop=True)
    
    print(f"\n{'='*60}")
    print(f"  TYPE FILTER: {TRAIN_TYPES}")
    print(f"  {_before_type} -> {len(_train_pdf)} rows")
    print(f"{'='*60}")
    for _t in sorted(_train_pdf['_type'].unique()):
        print(f"    {_t}: {(_train_pdf['_type']==_t).sum()}")
    
    _train_pdf = _train_pdf.drop(columns=['_type'])
    train_df = pl.from_pandas(_train_pdf)
else:
    print("\nType filter: DISABLED (all types)")
'''

with open(NB_PATH, 'r') as f:
    nb = json.load(f)

# Convert source string to notebook format (list of lines ending with \n)
lines = NEW_SOURCE.split('\n')
source_lines = []
for i, line in enumerate(lines):
    if i < len(lines) - 1:
        source_lines.append(line + '\n')
    else:
        if line:  # last line only if non-empty
            source_lines.append(line)

nb['cells'][6]['source'] = source_lines

with open(NB_PATH, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

# Verify
with open(NB_PATH, 'r') as f:
    nb2 = json.load(f)
src = ''.join(nb2['cells'][6]['source'])
print(f"Cell 7 updated successfully!")
print(f"  V84: {'V84' in src}")
print(f"  TYPE FILTER: {'TYPE FILTER' in src}")
print(f"  _infer_type: {'def _infer_type' in src}")
print(f"  Lines: {len(src.split(chr(10)))}")
print(f"  First line: {src.split(chr(10))[0]}")
# Check no garbled lines
for i, line in enumerate(src.split('\n')):
    if '    ' in line and line.strip().startswith('return') and '    ' in line[line.index('return')+6:]:
        print(f"  WARNING: Possible garbled line {i}: {line[:80]}")
        break
else:
    print(f"  No garbled lines detected ✅")
