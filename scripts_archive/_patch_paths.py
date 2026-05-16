"""Fix data paths to handle both old and new Kaggle mount styles."""
import json

NB_PATH = "nvidia-nemotron-2stage-sft.ipynb"

with open(NB_PATH) as f:
    nb = json.load(f)

# Cell nb_idx=6 is the data loading cell
src = ''.join(nb['cells'][6]['source'])

# Replace hardcoded COT_DATA path with auto-detection
old_block = """COMP_DATA = '/kaggle/input/nvidia-nemotron-3-reasoning-challenge'
COT_DATA = '/kaggle/input/prog-cot-training-data'

# Debug: list available files
print('Files in COT_DATA:', os.listdir(COT_DATA) if os.path.isdir(COT_DATA) else 'DIR NOT FOUND')
for d in ['/kaggle/input']:
    if os.path.isdir(d):
        for sub in os.listdir(d):
            full = os.path.join(d, sub)
            if os.path.isdir(full):
                files = os.listdir(full)
                print(f'  {sub}/: {files[:10]}')

# Load the merged dataset
train_df = pl.read_csv(f'{COT_DATA}/sft_merged_v1.csv')"""

new_block = """# Auto-detect data paths (Kaggle mounts datasets differently depending on config)
_CANDIDATES = [
    '/kaggle/input/prog-cot-training-data',
    '/kaggle/input/datasets/hastws/prog-cot-training-data',
    '/kaggle/input/datasets/hastws/prog-cot-training-data/versions/default',
]
COT_DATA = None
for _c in _CANDIDATES:
    if os.path.isdir(_c):
        COT_DATA = _c
        break
if COT_DATA is None:
    # Fallback: search for sft_merged_v1.csv anywhere under /kaggle/input
    import glob as _glob
    hits = _glob.glob('/kaggle/input/**/sft_merged_v1.csv', recursive=True)
    if hits:
        COT_DATA = os.path.dirname(hits[0])
    else:
        # List what's available for debugging
        for root, dirs, files in os.walk('/kaggle/input'):
            if files:
                print(f"  {root}: {files[:5]}")
            if root.count(os.sep) > 5:
                break
        raise FileNotFoundError("Cannot find sft_merged_v1.csv under /kaggle/input/")
print(f"COT_DATA = {COT_DATA}")
print(f"  Files: {os.listdir(COT_DATA)[:10]}")

COMP_DATA = '/kaggle/input/nvidia-nemotron-model-reasoning-challenge'
if not os.path.isdir(COMP_DATA):
    for _c in ['/kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge',
               '/kaggle/input/nvidia-nemotron-3-reasoning-challenge']:
        if os.path.isdir(_c):
            COMP_DATA = _c
            break

# Load the merged dataset
train_df = pl.read_csv(f'{COT_DATA}/sft_merged_v1.csv')"""

assert old_block in src, f"Old block not found! Current src starts with: {src[:200]}"
src = src.replace(old_block, new_block)

nb['cells'][6]['source'] = [line + '\n' for line in src.split('\n')]
nb['cells'][6]['source'][-1] = nb['cells'][6]['source'][-1].rstrip('\n')

with open(NB_PATH, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

src2 = ''.join(nb['cells'][6]['source'])
assert '_CANDIDATES' in src2
assert 'Auto-detect' in src2
print("OK: Data loading cell patched with auto-detection")
