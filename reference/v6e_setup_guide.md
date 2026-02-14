# TPU v6e Setup Guide - Complete Working Steps


### Setup
```bash
sudo timedatectl set-timezone Asia/Kolkata
export WANDB_API_KEY=$(gsutil cat gs://penrose_diffusion/wandb_api_key.txt)
echo "set -g mouse on" >> ~/.tmux.conf && tmux source-file ~/.tmux.conf

# Kill and permanently disable auto-upgrades
sudo systemctl stop unattended-upgrades apt-daily apt-daily-upgrade
sudo systemctl disable unattended-upgrades apt-daily apt-daily-upgrade
sudo systemctl mask unattended-upgrades apt-daily apt-daily-upgrade

# Lock-safe apt
while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do
  sleep 2
done

sudo NEEDRESTART_MODE=a apt update
sudo NEEDRESTART_MODE=a apt install -y python3-venv
python3 -m venv ~/tpu-env
source ~/tpu-env/bin/activate

pip install --upgrade pip
pip install fsspec gcsfs tqdm scipy wandb  --no-warn-script-location
pip install torch~=2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch_xla[tpu]~=2.6.0 \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html

git clone https://github.com/rakeshvar/penrose_diffusion
cd penrose_diffusion
mkdir datasets/
gcloud storage cp -r "gs://penrose_diffusion/datasets/*.npz" datasets/
```

### Train
```bash
python train.py datasets/hex_t096_c96_u18.npz ...
```

### Kill jobs
```bash
sudo fuser -v /dev/vfio/*
pkill -9 -f python
sudo fuser -v /dev/vfio/*
```

### Toys
```bash
python train.py dd32 toy datasets/hexxy_t096_c01_u18.npz
python train.py is32 toy datasets/hexxy_t096_c01_u18.npz
python train.py ld32 toy datasets/hexxy_t096_c01_u18.npz
python train.py ll32 toy datasets/hexind_t128_c01_u16.npz
python train.py di32 toy datasets/hexind_t128_c01_u16.npz
```

# Info

## Zone
```bash
echo "Zone: $(curl -s http://metadata.google.internal/computeMetadata/v1/instance/zone -H "Metadata-Flavor: Google" | cut -d/ -f4)"
```

## bf16
```
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump" XLA_USE_BF16=0 XLA_DOWNCAST_BF16=0  python train.py ld32 toy w0 ne3 s0 datasets/hex_t096_c16_u18.npz 
```

## tmux
| Action | Command / Key Stroke |
| :--- | :--- |
| **Start Named Session** | `tmux new -s <name>` |
| **Detach from Session** | `Ctrl + b` then `d` |
| **List Active Sessions** | `tmux ls` |
| **Reattach to Session** | `tmux attach -t <name>` |
| **Kill/Close Session** | `tmux kill-session -t <name>` |
| **Split Vertically** | `Ctrl + b` then `%` |
| **Split Horizontally** | `Ctrl + b` then `"` |
| **Switch Panes** | `Ctrl + b` then `Arrow Keys` |
| **Scroll/Copy Mode** | `Ctrl + b` then `[` (Press `q` to exit) |

## Monitor

```
pip install tpu-info
tpu-info -s 1
# OR
pip install tpubar
tpubar monitor
```