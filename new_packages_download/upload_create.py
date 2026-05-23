"""
Step 1: Create the dataset with only small files (dir_mode='skip' ignores subdirectories).
This finishes in seconds and makes the dataset exist on Kaggle.

Step 2: Run upload_version.py to upload the large custom_cuda/ and official/ zips
as a new version.
"""
import sys
import kaggle

api = kaggle.KaggleApi()
api.authenticate()

folder = "."

print("Step 1: Creating dataset with small files only (skipping subdirectories)...")
result = api.dataset_create_new(
    folder,
    public=False,
    quiet=False,
    dir_mode="skip",  # skip custom_cuda/ and official/ — add them via version later
)

if result is None:
    print("ERROR: dataset_create_new returned None", file=sys.stderr)
    sys.exit(1)
elif result.error:
    print("ERROR:", result.error, file=sys.stderr)
    sys.exit(1)
elif result.status and result.status.lower() == "ok":
    print("Success! Dataset is being created at:", result.url)
else:
    print("Unexpected result:", result)
