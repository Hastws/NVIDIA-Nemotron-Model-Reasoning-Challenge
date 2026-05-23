"""
Re-download failed nvidia-* packages by querying the PyPI JSON API directly
and using urllib to download x86_64 Linux wheels.
Bypasses pip's platform compatibility checking entirely.
"""
from pathlib import Path
import urllib.request, json

DEST = Path("official")
DEST.mkdir(exist_ok=True)

PYPI_BASE = "https://pypi.org/pypi"

# Packages that still failed after first retry
failed_pkgs = [
    ("nvidia-cuda-cupti", "13.0.85"),
    ("nvidia-cuda-cupti-cu12", "12.8.90"),
    ("nvidia-cuda-nvrtc", "13.0.88"),
    ("nvidia-cuda-nvrtc-cu12", "12.8.93"),
    ("nvidia-cuda-runtime", "13.0.96"),
    ("nvidia-cuda-runtime-cu12", "12.8.90"),
    ("nvidia-cufft", "12.0.0.61"),
    ("nvidia-cufft-cu12", "11.3.3.83"),
    ("nvidia-cufile", "1.15.1.6"),
    ("nvidia-cufile-cu12", "1.13.1.3"),
    ("nvidia-cusparse", "12.6.3.3"),
    ("nvidia-cusparse-cu12", "12.5.8.93"),
    ("nvidia-cusparselt-cu12", "0.7.1"),
    ("nvidia-cusparselt-cu13", "0.8.0"),
    ("nvidia-nccl-cu12", "2.27.5"),   # version may not exist; try latest
    ("nvidia-nccl-cu13", "2.28.9"),   # version may not exist; try latest
    ("nvidia-nvjitlink", "13.0.88"),
    ("nvidia-nvjitlink-cu12", "12.8.93"),
    ("nvidia-nvshmem-cu12", "3.4.5"),
    ("nvidia-nvshmem-cu13", "3.4.5"),
    ("nvidia-nvtx", "13.0.85"),
    ("nvidia-nvtx-cu12", "12.8.90"),
]

def get_wheel_url(name, version):
    """Query PyPI JSON API and return (url, filename) for x86_64 Linux wheel."""
    url = f"{PYPI_BASE}/{name}/{version}/json"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read())
    except Exception:
        return None, None
    for f in data.get("urls", []):
        fname = f["filename"]
        if fname.endswith(".whl") and "x86_64" in fname and "win" not in fname and "aarch" not in fname:
            return f["url"], fname
    return None, None

def get_latest_x86_wheel(name):
    """Try the latest release if exact version is not on PyPI."""
    url = f"{PYPI_BASE}/{name}/json"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read())
    except Exception:
        return None, None, None
    latest = data["info"]["version"]
    wheel_url, fname = get_wheel_url(name, latest)
    return wheel_url, fname, latest

ok, skip, fail = [], [], []

for name, version in failed_pkgs:
    # Skip if already downloaded
    existing = list(DEST.glob(f"{name.replace('-','_')}-{version}*x86_64*.whl"))
    if existing:
        print(f"  SKIP (exists): {existing[0].name}")
        ok.append(f"{name}=={version}")
        continue

    print(f"[?] {name}=={version} ... ", end="", flush=True)
    wheel_url, fname = get_wheel_url(name, version)
    used_ver = version

    if not wheel_url:
        wheel_url, fname, used_ver = get_latest_x86_wheel(name)
        if not wheel_url:
            print("✗  not on PyPI (system CUDA library — Kaggle provides it)")
            skip.append(f"{name}=={version}")
            continue
        print(f"\n    version {version} not on PyPI → using latest {used_ver}")
        print(f"    {fname} ... ", end="", flush=True)

    dest_file = DEST / fname
    try:
        urllib.request.urlretrieve(wheel_url, dest_file)
        print(f"✓")
        ok.append(f"{name}=={used_ver}")
    except Exception as e:
        print(f"✗  {e}")
        fail.append(f"{name}=={version}")

print(f"\n完成: 成功 {len(ok)} / 不在PyPI(跳过) {len(skip)} / 失败 {len(fail)}")
if skip:
    print("\n这些是CUDA系统库，Kaggle Docker镜像已自带，无需离线包:")
    for p in skip:
        print(f"  {p}")
if fail:
    print("\n下载失败:")
    for p in fail:
        print(f"  {p}")
