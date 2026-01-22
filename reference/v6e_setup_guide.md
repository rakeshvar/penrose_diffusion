# TPU v6e Setup Guide - Complete Working Steps

## Enter

```bash
gcloud compute tpus tpu-vm create penrose-train \
  --zone=europe-west4-a \
  --accelerator-type=v6e-8 \
  --version=v2-alpha-tpuv6e \
  --preemptible
```


```bash
gcloud compute tpus tpu-vm ssh penrose-train --zone=europe-west4-a
```

## Python & Venv & pip

```bash
python3 --version
```

Expected output: `Python 3.10.6`

### Virtual Environment
```bash
sudo NEEDRESTART_MODE=a apt update
sudo NEEDRESTART_MODE=a apt install -y python3.10-venv
python3 -m venv ~/tpu-env
source ~/tpu-env/bin/activate```

### pip
```bash
pip install --upgrade pip
#pip uninstall -y torch torch-xla torchvision torchaudio
pip install fsspec gcsfs 
pip install tqdm scipy wandb
pip install torch~=2.6.0 torch_xla[tpu]~=2.6.0 \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html
```
*torch cpu version!*

## Test
### [Optional] Test TPU Detection


```bash
PJRT_DEVICE=TPU python3 -c "import torch_xla.core.xla_model as xm; print('TPU device:', xm.xla_device())"
```

Expected output:
```
TPU device: xla:0
```


### [Optional] Test Computation


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
### Verify saving to gs
```bash
python3 << 'EOF'
pass
EOF
```

###  PJRT_DEVICE 

```bash
export PJRT_DEVICE=TPU
```

## Training
### Clone your repo
```
git clone https://github.com/rakeshvar/penrose_diffusion
cd penrose_diffusion
```

### [Optional] Copy Datasets
```bash
mkdir -p datasets
gsutil -m cp gs://penrose_diffusion/datasets/*.npz datasets
```

### Train
```bash
python train.py gs://penrose_diffusion/datasets/hex_t096_c96_u18.npz -t num_epochs=10
```

