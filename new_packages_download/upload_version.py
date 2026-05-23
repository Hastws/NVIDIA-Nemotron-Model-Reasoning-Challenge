"""
Step 2: Upload large directories (custom_cuda/, official/) as a new dataset version.
Run this AFTER upload_create.py has successfully created the dataset.
"""
import sys
import kaggle

api = kaggle.KaggleApi()
api.authenticate()

folder = "."

print("Step 2: Uploading large zip files as new dataset version...")
result = api.dataset_create_version(
    folder,
    version_notes="add custom_cuda and official packages",
    quiet=False,
    dir_mode="zip",  # zips custom_cuda/ -> custom_cuda.zip, official/ -> official.zip
)

if result is None:
    print("ERROR: dataset_create_version returned None", file=sys.stderr)
    sys.exit(1)
elif result.error:
    print("ERROR:", result.error, file=sys.stderr)
    sys.exit(1)
elif result.status and result.status.lower() == "ok":
    print("Success! New version is being created at:", result.url)
else:
    print("Unexpected result:", result)
