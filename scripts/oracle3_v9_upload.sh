#!/usr/bin/env bash
# =============================================================================
# Run on oracle3 (129.153.71.165) to:
#   1. Download submission.zip from Kaggle kernel output
#   2. Extract adapter, fix config
#   3. Upload as new Kaggle dataset: hastws/nemotron-v9
# =============================================================================
set -euo pipefail

WORKDIR=~/work_space/nemotron/v9_upload
mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "=== Step 1: Download kernel output ==="
kaggle kernels output hastws/training-with-unsloth-to-achieve-0-85-no-lm-head -p "$WORKDIR"

echo ""
echo "=== Step 2: Find and extract submission.zip ==="
SUBMISSION_ZIP=$(find "$WORKDIR" -name "submission.zip" -type f | head -1)
if [ -z "$SUBMISSION_ZIP" ]; then
    echo "ERROR: submission.zip not found!"
    echo "Files in $WORKDIR:"
    find "$WORKDIR" -type f
    exit 1
fi
echo "Found: $SUBMISSION_ZIP"

mkdir -p "$WORKDIR/adapter"
unzip -o "$SUBMISSION_ZIP" -d "$WORKDIR/adapter"

# Find the actual adapter directory (containing adapter_config.json)
ADAPTER_DIR=$(dirname "$(find "$WORKDIR/adapter" -name "adapter_config.json" -type f | head -1)")
echo "Adapter directory: $ADAPTER_DIR"

echo ""
echo "=== Step 3: Fix adapter_config.json ==="
python3 -c "
import json
config_path = '$ADAPTER_DIR/adapter_config.json'
with open(config_path) as f:
    cfg = json.load(f)
cfg['base_model_name_or_path'] = 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'
cfg['inference_mode'] = True
cfg['lora_dropout'] = 0.0
with open(config_path, 'w') as f:
    json.dump(cfg, f, indent=2)
print('adapter_config.json updated for inference')
"

echo ""
echo "=== Step 4: Prepare dataset directory ==="
DATASET_DIR="$WORKDIR/dataset"
mkdir -p "$DATASET_DIR"
cp "$ADAPTER_DIR/adapter_config.json" "$DATASET_DIR/"
cp "$ADAPTER_DIR/adapter_model.safetensors" "$DATASET_DIR/"

cat > "$DATASET_DIR/dataset-metadata.json" << 'EOF'
{
  "title": "Nemotron v9 - LoRA SFT Adapter",
  "id": "hastws/nemotron-v9",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

echo "Dataset directory ready:"
ls -lh "$DATASET_DIR/"

echo ""
echo "=== Step 5: Upload to Kaggle ==="
# Try creating new version (dataset should already exist)
kaggle datasets version \
    -p "$DATASET_DIR" \
    -m "v9: LoRA SFT adapter from training-with-unsloth (no-lm-head), for GRPO reinforcement learning" \
    --dir-mode tar \
    2>&1 || {
    echo "Version create failed, trying dataset create..."
    kaggle datasets create -p "$DATASET_DIR" --dir-mode tar
}

echo ""
echo "=== DONE ==="
echo "Dataset: https://www.kaggle.com/datasets/hastws/nemotron-v9"
