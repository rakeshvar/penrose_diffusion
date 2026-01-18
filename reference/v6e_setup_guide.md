# TPU v6e Setup Guide - Complete Working Steps

## Prerequisites
You need a TPU v6e created with the correct runtime version.

## Step 1: Create TPU v6e with Ubuntu 24.04 Image

From your local machine, create the TPU:

```bash
gcloud compute tpus tpu-vm create penrose-train \
  --zone=europe-west4-a \
  --accelerator-type=v6e-8 \
  --version=v6e-ubuntu-2404 \
  --preemptible
```

**Key Point:** The `v6e-ubuntu-2404` runtime is specifically designed for v6e TPUs and includes proper PJRT support.

## Step 2: SSH into the TPU VM

```bash
gcloud compute tpus tpu-vm ssh penrose-train --zone=europe-west4-a
```

## Step 3: Verify Environment

Check Python version (should be 3.12.x):

```bash
python3 --version
```

Expected output: `Python 3.12.3`

## Step 4: Install Python Virtual Environment Package

Ubuntu 24.04 requires the venv package to be installed separately:

```bash
sudo apt update
sudo apt install -y python3.12-venv
```

## Step 5: Create Virtual Environment

Create and activate a Python virtual environment:

```bash
python3 -m venv ~/tpu-env
source ~/tpu-env/bin/activate
```

**Note:** Your prompt should now show `(tpu-env)` prefix.

Install requirements
```
pip install tqdm scipy
```

## Step 6: Install PyTorch (CPU version)

Install CPU-only PyTorch to avoid unnecessary CUDA dependencies:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Why CPU version?** TPUs don't need CUDA. The CPU version is smaller and faster to install.

## Step 7: Install PyTorch/XLA with TPU Support

Install PyTorch/XLA with the TPU plugin:

```bash
pip install torch-xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
```

This installs:
- `torch-xla`: XLA compiler integration for PyTorch
- TPU plugin and libtpu library
- All necessary dependencies

## Step 8: Test TPU Detection

Verify the TPU is detected:

```bash
PJRT_DEVICE=TPU python3 -c "import torch_xla.core.xla_model as xm; print('TPU device:', xm.xla_device())"
```

Expected output:
```
<string>:1: DeprecationWarning: Use torch_xla.device instead
TPU device: xla:0
```

**Success!** `xla:0` confirms the TPU is detected.

## Step 9: Run Test Computation

Verify the TPU can actually execute computations:

```bash
PJRT_DEVICE=TPU python3 << 'EOF'
import torch
import torch_xla.core.xla_model as xm

device = xm.xla_device()
print(f"Device: {device}")

# Simple computation on TPU
x = torch.randn(3, 3).to(device)
y = torch.randn(3, 3).to(device)
z = x + y

xm.mark_step()  # Execute the computation
print(f"Result shape: {z.shape}")
print("✅ TPU computation successful!")
EOF
```

Expected output:
```
Device: xla:0
Result shape: torch.Size([3, 3])
✅ TPU computation successful!
```

## Step 10: Make PJRT_DEVICE Permanent (Optional)

To avoid typing `PJRT_DEVICE=TPU` every time, add it to your shell profile:

```bash
echo 'export PJRT_DEVICE=TPU' >> ~/.bashrc
source ~/.bashrc
```

Now you can run Python scripts without the prefix:

```bash
python3 your_script.py  # TPU will be used automatically
```

## Clone your repo
```
git clone https://github.com/rakeshvar/penrose_diffusion
cd penrose_diffusion
```

---

## Summary

### Working Configuration
- **TPU Type:** v6e-8
- **Runtime Version:** `v6e-ubuntu-2404`
- **Python:** 3.12.3
- **PyTorch:** Latest CPU version
- **PyTorch/XLA:** Latest from libtpu-releases
- **Required Env Var:** `PJRT_DEVICE=TPU`

### Key Takeaways
1. **v6e requires PJRT** - XRT is not supported
2. **Use v6e-ubuntu-2404 runtime** - Other runtimes may not have proper drivers
3. **Virtual environment is required** - Python 3.12 enforces this
4. **CPU PyTorch is sufficient** - TPUs don't need CUDA
5. **Set PJRT_DEVICE=TPU** - This tells PyTorch/XLA to use the TPU

### Troubleshooting
If TPU is not detected:
- Verify `PJRT_DEVICE=TPU` is set
- Check you're in the virtual environment (`(tpu-env)` in prompt)
- Ensure you used `v6e-ubuntu-2404` runtime when creating the TPU
- Try `pip install --upgrade torch-xla[tpu]`