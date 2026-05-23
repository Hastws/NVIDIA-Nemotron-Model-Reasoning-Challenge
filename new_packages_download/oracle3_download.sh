#!/usr/bin/env bash
# Run on oracle3 (US server) to download all packages for Linux x86_64 / Python 3.12
# Strict x86_64-only: never falls back to host-arch (oracle3 is aarch64)

WORKDIR=~/work_space/nemotron/new_packages_download
cd "$WORKDIR"

DEST=official
mkdir -p "$DEST"

# Include all manylinux platform variants needed by modern packages:
#   manylinux2014 = manylinux_2_17 (torch, numpy use 2_24/2_27/2_28/2_31)
# Do NOT specify --abi so pip auto-generates cp312, abi3, none compat tags
PLATFORM_OPTS="
  --platform manylinux2014_x86_64
  --platform linux_x86_64
  --platform manylinux_2_17_x86_64
  --platform manylinux_2_24_x86_64
  --platform manylinux_2_27_x86_64
  --platform manylinux_2_28_x86_64
  --platform manylinux_2_31_x86_64
  --platform manylinux_2_34_x86_64
  --platform manylinux_2_35_x86_64"
PY_OPTS="--python-version 312 --implementation cp --only-binary :all:"

failed_pkgs=()
success_count=0

echo "=== Step 1: pip download official packages (per-pkg, strict x86_64) ==="
while IFS= read -r line; do
  # Skip empty lines and comments
  [[ -z "$line" || "$line" == \#* ]] && continue
  pkg="$line"
  pkg_basename=$(echo "$pkg" | sed 's/[=<>!].*//' | tr '-' '_' | tr '[:upper:]' '[:lower:]')

  # Skip if already downloaded (any version/platform file matching the name)
  if compgen -G "$DEST/${pkg_basename}-*" > /dev/null 2>&1; then
    echo "SKIP (exists): $pkg"
    ((success_count++))
    continue
  fi

  echo "--- $pkg ---"
  if pip3 download $PLATFORM_OPTS $PY_OPTS -d "$DEST" "$pkg" --no-deps 2>&1; then
    ((success_count++))
  else
    failed_pkgs+=("$pkg")
    echo "FAILED: $pkg"
  fi
done < requirements.txt

echo ""
echo "=== Step 1 Summary ==="
echo "  Success: $success_count / $(grep -c '^[^#]' requirements.txt) packages"
echo "  Failed:  ${#failed_pkgs[@]}"
for f in "${failed_pkgs[@]}"; do echo "    FAILED: $f"; done

echo ""
echo "=== Step 2: Download missing nvidia packages directly (x86_64) ==="
python3 download_nvidia_direct.py

echo ""
echo "=== Download complete ==="
echo "official/: $(ls official/*.whl 2>/dev/null | wc -l) whl files"
echo "custom_cuda/: $(ls custom_cuda/*.whl 2>/dev/null | wc -l) whl files"
