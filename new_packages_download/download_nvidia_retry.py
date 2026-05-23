"""
Re-download the 30 nvidia-* packages that failed because the previous script
used --platform manylinux_2_28_x86_64 but NVIDIA wheels use manylinux_2_27_x86_64.
"""
import subprocess, sys
from pathlib import Path

DEST = Path("official")
DEST.mkdir(exist_ok=True)

failed_pkgs = [
    "nvidia-cublas==13.1.0.3",
    "nvidia-cublas-cu12==12.8.4.1",
    "nvidia-cuda-cupti==13.0.85",
    "nvidia-cuda-cupti-cu12==12.8.90",
    "nvidia-cuda-nvrtc==13.0.88",
    "nvidia-cuda-nvrtc-cu12==12.8.93",
    "nvidia-cuda-runtime==13.0.96",
    "nvidia-cuda-runtime-cu12==12.8.90",
    "nvidia-cudnn-cu12==9.10.2.21",
    "nvidia-cudnn-cu13==9.19.0.56",
    "nvidia-cufft==12.0.0.61",
    "nvidia-cufft-cu12==11.3.3.83",
    "nvidia-cufile==1.15.1.6",
    "nvidia-cufile-cu12==1.13.1.3",
    "nvidia-curand==10.4.0.35",
    "nvidia-curand-cu12==10.3.9.90",
    "nvidia-cusolver==12.0.4.66",
    "nvidia-cusolver-cu12==11.7.3.90",
    "nvidia-cusparse==12.6.3.3",
    "nvidia-cusparse-cu12==12.5.8.93",
    "nvidia-cusparselt-cu12==0.7.1",
    "nvidia-cusparselt-cu13==0.8.0",
    "nvidia-nccl-cu12==2.27.5",
    "nvidia-nccl-cu13==2.28.9",
    "nvidia-nvjitlink==13.0.88",
    "nvidia-nvjitlink-cu12==12.8.93",
    "nvidia-nvshmem-cu12==3.4.5",
    "nvidia-nvshmem-cu13==3.4.5",
    "nvidia-nvtx==13.0.85",
    "nvidia-nvtx-cu12==12.8.90",
]

ok, fail = [], []

PYPI = "https://pypi.org/simple/"

for i, pkg in enumerate(failed_pkgs, 1):
    print(f"[{i:2d}/{len(failed_pkgs)}] {pkg} ... ", end="", flush=True)
    # First try: manylinux_2_27_x86_64 from official PyPI
    r = subprocess.run(
        [sys.executable, "-m", "pip", "download",
         "--no-deps",
         "--index-url", PYPI,
         "--platform", "manylinux_2_27_x86_64",
         "--python-version", "312",
         "--implementation", "cp",
         "--only-binary", ":all:",
         "-d", str(DEST),
         pkg],
        capture_output=True, text=True,
    )
    if r.returncode == 0:
        print("✓")
        ok.append(pkg)
        continue
    # Second try: no platform restriction from official PyPI
    r2 = subprocess.run(
        [sys.executable, "-m", "pip", "download",
         "--no-deps",
         "--index-url", PYPI,
         "-d", str(DEST), pkg],
        capture_output=True, text=True,
    )
    if r2.returncode == 0:
        print("✓ (any-platform)")
        ok.append(pkg)
    else:
        err = (r2.stderr or r.stderr).strip().splitlines()[-1]
        print(f"✗  {err[:80]}")
        fail.append((pkg, err))

print(f"\n完成: 成功 {len(ok)} / 失败 {len(fail)}")
if fail:
    print("\n仍然失败的包:")
    for p, e in fail:
        print(f"  {p}\n    {e}")
