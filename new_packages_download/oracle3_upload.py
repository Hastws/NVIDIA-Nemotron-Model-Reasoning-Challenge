"""
Upload dataset version to Kaggle from oracle3 (US server, fast upload).
Uses dir_mode='zip' so Kaggle receives official/ and custom_cuda/ as directories
(Kaggle will extract them, giving official/*.whl structure in the dataset).
"""
import sys, os
from pathlib import Path
import kaggle

HERE = Path(__file__).parent

# Verify we have the packages
official_count = len(list((HERE / "official").glob("*.whl")))
custom_count   = len(list((HERE / "custom_cuda").glob("*.whl")))
print(f"official/*.whl:    {official_count} files")
print(f"custom_cuda/*.whl: {custom_count} files")

if official_count == 0 or custom_count == 0:
    print("ERROR: Missing packages. Run oracle3_download.sh first.", file=sys.stderr)
    sys.exit(1)

print("\nUploading new dataset version (dir_mode=zip)...")
api = kaggle.KaggleApi()
api.authenticate()

result = api.dataset_create_version(
    str(HERE),
    version_notes=f"v4: {official_count} official wheels + {custom_count} custom_cuda wheels (uploaded from US server)",
    quiet=False,
    dir_mode="zip",        # zip each subdir (official/, custom_cuda/); Kaggle extracts → official/*.whl
    delete_old_versions=False,
)

if result is None:
    print("ERROR: dataset_create_version returned None", file=sys.stderr)
    sys.exit(1)
elif hasattr(result, 'error') and result.error:
    print("ERROR:", result.error, file=sys.stderr)
    sys.exit(1)
else:
    print("Success!", result)
