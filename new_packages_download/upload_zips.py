"""
Step 3: Pre-create zip files locally, then upload them as a new dataset version.
Using dir_mode='skip' so Kaggle treats the pre-built zips as plain files.
"""
import sys, zipfile, os
from pathlib import Path
import kaggle

HERE = Path(__file__).parent

def make_zip(src_dir: Path, out_zip: Path):
    """Zip a directory, showing progress."""
    files = sorted(src_dir.rglob("*"))
    files = [f for f in files if f.is_file()]
    total = len(files)
    print(f"  Zipping {total} files from {src_dir.name}/ → {out_zip.name} ...")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for i, f in enumerate(files, 1):
            zf.write(f, f.relative_to(src_dir))
            if i % 20 == 0 or i == total:
                print(f"    {i}/{total}", end="\r", flush=True)
    size_gb = out_zip.stat().st_size / 1e9
    print(f"  Done: {out_zip.name} ({size_gb:.2f} GB)        ")

# Build zips if not already present
official_zip = HERE / "official.zip"
custom_zip   = HERE / "custom_cuda.zip"

if not official_zip.exists():
    make_zip(HERE / "official", official_zip)
else:
    print(f"  official.zip already exists ({official_zip.stat().st_size/1e9:.2f} GB), skipping zip")

if not custom_zip.exists():
    make_zip(HERE / "custom_cuda", custom_zip)
else:
    print(f"  custom_cuda.zip already exists ({custom_zip.stat().st_size/1e9:.2f} GB), skipping zip")

# Upload as new version
print("\nUploading new dataset version (dir_mode=skip)...")
api = kaggle.KaggleApi()
api.authenticate()

result = api.dataset_create_version(
    str(HERE),
    version_notes="add custom_cuda.zip and official.zip (pre-built)",
    quiet=False,
    dir_mode="skip",   # upload files in root, skip subdirectories
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
