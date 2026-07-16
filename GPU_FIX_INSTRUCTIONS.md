# GPU Fix Instructions for RTX 5050 (sm_120)

## Problem
RTX 5050 has CUDA compute capability sm_120 (Blackwell)
Current PyTorch stable only supports up to sm_90 (Hopper)

## Solution: Install PyTorch Nightly

### Step 1: Uninstall current PyTorch
pip uninstall torch torchvision torchaudio -y

### Step 2: Install PyTorch Nightly with CUDA 12.4
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

### Step 3: Verify GPU detection
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU Count: {torch.cuda.device_count()}')"

## Expected Output
CUDA Available: True
GPU Name: NVIDIA GeForce RTX 5050 Laptop GPU
GPU Count: 1

## Benefits After Fix
- BERT training: ~20-30 min → 2-3 min (10x faster!)
- GPU memory: 8GB VRAM available
- Better performance for all PyTorch operations

## Note
Nightly builds are experimental but generally stable.
If you encounter issues, you can always reinstall stable CPU version.
