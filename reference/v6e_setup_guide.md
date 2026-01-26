# TPU v6e Setup Guide - Complete Working Steps


### Setup
```bash
sudo timedatectl set-timezone Asia/Kolkata
export WANDB_API_KEY=$(gsutil cat gs://penrose_diffusion/wandb_api_key.txt)
sudo NEEDRESTART_MODE=a apt update
sudo NEEDRESTART_MODE=a apt install -y python3.10-venv
python3 -m venv ~/tpu-env
source ~/tpu-env/bin/activate

pip install --upgrade pip
pip install fsspec gcsfs
pip install tqdm scipy wandb
pip install torch~=2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch_xla[tpu]~=2.6.0 \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html
```

```
pip install torch~=2.6.0 torch_xla[tpu]~=2.6.0 \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html
```

### Clone
```
git clone https://github.com/rakeshvar/penrose_diffusion
cd penrose_diffusion
```

### Train
```bash
# toy 
python train.py gs://penrose_diffusion/datasets/hex_t096_c96_u18.npz toy -t num_epochs=3 -t loss=pil 
# main
python train.py gs://penrose_diffusion/datasets/hex_t096_c96_u18.npz model128 -t num_epochs=301 -t loss=pil 
```

### Kill jobs
```bash
sudo fuser -v /dev/vfio/*
pkill -9 -f python
sudo fuser -v /dev/vfio/*
```


# Working Area
```
/home/raka/.local/lib/python3.10/site-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
```