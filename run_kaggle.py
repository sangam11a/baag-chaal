"""
Kaggle GPU Automated Runner for Bagh-Chal AI Research Platform.

Automates:
1. Environment and CUDA GPU hardware inspection
2. Environment variable configuration
3. Test suite execution (65 unit and integration tests)
4. 80-game baseline tournament benchmark
5. Execution of all 5 Research Gap Experiments (RG1 - RG5)
6. Output artifact packaging into /kaggle/working/bagh_chal_results.zip
"""

import os
import sys
import time
import zipfile
import subprocess
from pathlib import Path
import torch

def setup_environment():
    print("=" * 70)
    print("  BAGH-CHAL AI: KAGGLE GPU ENVIRONMENT INITIALIZATION")
    print("=" * 70)

    # 1. Inspect Python and PyTorch
    print(f"Python Version : {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # 2. Inspect CUDA GPU Hardware
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available : {cuda_available}")

    if cuda_available:
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        total_mem = torch.cuda.get_device_properties(current_device).total_memory / (1024 ** 3)
        print(f"GPU Devices    : {device_count}")
        print(f"Active GPU     : [{current_device}] {device_name}")
        print(f"VRAM Available : {total_mem:.2f} GB")
        
        # Set default device env var
        os.environ.setdefault("BAGH_CHAL_DEVICE", "cuda:0" if device_count > 1 else "cuda")
    else:
        print("[WARNING] CUDA is not available! Running on CPU fallback.")
        os.environ.setdefault("BAGH_CHAL_DEVICE", "cpu")

    # Set key recommended environment variables if unset
    os.environ.setdefault("PYTHONPATH", ".")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("WANDB_MODE", "disabled")

    print("\nActive Environment Variables:")
    for var in [
        "BAGH_CHAL_DEVICE",
        "CUDA_VISIBLE_DEVICES",
        "PYTHONPATH",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "PYTORCH_CUDA_ALLOC_CONF",
    ]:
        print(f"  {var:<24} = {os.environ.get(var, '<unset>')}")
    print("=" * 70)


def run_tests():
    print("\n[Step 1/3] Running PyTest Suite (65 Unit & Integration Tests)...")
    t0 = time.time()
    res = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], capture_output=False)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"[ERROR] Test suite failed with exit code {res.returncode}")
        sys.exit(res.returncode)
    print(f"[SUCCESS] All tests passed in {elapsed:.2f}s.\n")


def run_tournament():
    print("[Step 2/3] Executing 80-Game Baseline Tournament...")
    t0 = time.time()
    from src.evaluation.run_benchmark import run_benchmark
    run_benchmark(rounds_per_matchup=5, base_seed=1000)
    elapsed = time.time() - t0
    print(f"[SUCCESS] Tournament completed in {elapsed:.2f}s.\n")


def run_research_gaps():
    print("[Step 3/3] Executing All 5 Research Gap Experiments (RG1 - RG5)...")
    t0 = time.time()
    from src.experiments.run_all_experiments import run_full_experimental_campaign
    run_full_experimental_campaign()
    elapsed = time.time() - t0
    print(f"[SUCCESS] All 5 Research Gap experiments completed in {elapsed:.2f}s.\n")


def package_results():
    print("\nPackaging Results & Artifacts for Download...")
    zip_filename = "bagh_chal_results.zip"
    target_dirs = ["data", "logs"]
    target_files = ["observation.md", "literature_review.md"]

    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for t_file in target_files:
            if os.path.exists(t_file):
                zipf.write(t_file, t_file)
                print(f"  + Added {t_file}")
        for t_dir in target_dirs:
            if os.path.exists(t_dir):
                for root, _, files in os.walk(t_dir):
                    for file in files:
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, ".")
                        zipf.write(full_path, rel_path)
                print(f"  + Added directory {t_dir}/")

    zip_size_mb = os.path.getsize(zip_filename) / (1024 * 1024)
    print(f"\n[DONE] Archive created: {zip_filename} ({zip_size_mb:.2f} MB)")
    print(f"You can now download '{zip_filename}' from the Kaggle Notebook Output pane!")


if __name__ == "__main__":
    setup_environment()
    run_tests()
    run_tournament()
    run_research_gaps()
    package_results()

