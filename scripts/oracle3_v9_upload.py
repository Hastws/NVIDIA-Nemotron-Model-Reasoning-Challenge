"""v9 pipeline: download kernel output, extract adapter, upload to Kaggle.
Run on oracle3: python3 oracle3_v9_upload.py"""
import os, sys, json, shutil, zipfile
from pathlib import Path

os.environ["KAGGLE_API_TOKEN"] = "KGAT_e6c5e593cd30c4346ee829b141a186e0"
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
print("Auth OK")

WORK_DIR = Path.home() / "work_space" / "nemotron" / "v9_upload"
WORK_DIR.mkdir(parents=True, exist_ok=True)

print("\n=== Step 1: Download kernel output ===")
kernel_ref = "hastws/training-with-unsloth-to-achieve-0-85-no-lm-head"
api.kernels_output(kernel_ref, path=str(WORK_DIR))
print("Downloaded to", WORK_DIR)

print("\n=== Step 2: Find submission.zip ===")
submission_zips = list(WORK_DIR.rglob("submission.zip"))
if not submission_zips:
    all_zips = list(WORK_DIR.rglob("*.zip"))
    print("All zips:", all_zips)
    submission_zip = all_zips[0] if all_zips else None
    if not submission_zip:
        print("ERROR: No zip found. Contents:", list(WORK_DIR.iterdir()))
        sys.exit(1)
else:
    submission_zip = submission_zips[0]
sz = submission_zip.stat().st_size / 1024 / 1024
print(f"Found: {submission_zip} ({sz:.1f} MB)")

print("\n=== Step 3: Extract adapter ===")
EXTRACT_DIR = WORK_DIR / "adapter"
EXTRACT_DIR.mkdir(exist_ok=True)
with zipfile.ZipFile(submission_zip, "r") as zf:
    zf.extractall(EXTRACT_DIR)

adapter_files = list(EXTRACT_DIR.rglob("adapter_config.json"))
if not adapter_files:
    print("ERROR: adapter_config.json not found!")
    sys.exit(1)
adapter_dir = adapter_files[0].parent
print(f"Adapter dir: {adapter_dir}")
for f in adapter_dir.iterdir():
    print(f"  {f.name} ({f.stat().st_size/1024/1024:.1f} MB)")

print("\n=== Step 4: Fix adapter_config.json ===")
config_path = adapter_dir / "adapter_config.json"
with open(config_path) as f:
    cfg = json.load(f)
cfg["base_model_name_or_path"] = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
cfg["inference_mode"] = True
cfg["lora_dropout"] = 0.0
with open(config_path, "w") as f:
    json.dump(cfg, f, indent=2)
print("Config: r={}".format(cfg.get("r", "?")))

print("\n=== Step 5: Prepare dataset dir ===")
DATASET_DIR = WORK_DIR / "dataset"
DATASET_DIR.mkdir(exist_ok=True)
shutil.copy2(adapter_dir / "adapter_config.json", DATASET_DIR / "adapter_config.json")
shutil.copy2(adapter_dir / "adapter_model.safetensors", DATASET_DIR / "adapter_model.safetensors")

metadata = {
    "title": "Nemotron v9 - LoRA SFT Adapter",
    "id": "hastws/nemotron-v9",
    "licenses": [{"name": "CC0-1.0"}]
}
with open(DATASET_DIR / "dataset-metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("Files:", [f.name for f in DATASET_DIR.iterdir()])

print("\n=== Step 6: Upload to Kaggle ===")
try:
    api.dataset_create_version(
        folder=str(DATASET_DIR),
        version_notes="v9: LoRA SFT adapter from training-with-unsloth (no-lm-head), for GRPO RL",
        quiet=False, dir_mode="tar", delete_old_versions=False)
    print("SUCCESS!")
except Exception as e:
    err = str(e)
    print("Version failed: {}".format(err[:200]))
    print("Trying create new dataset...")
    try:
        api.dataset_create_new(folder=str(DATASET_DIR), public=True, dir_mode="tar")
        print("Created new dataset!")
    except Exception as e2:
        print("Create also failed: {}".format(e2))
        sys.exit(1)

print("\n=== DONE ===")
print("Dataset: https://www.kaggle.com/datasets/hastws/nemotron-v9")
