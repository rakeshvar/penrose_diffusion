# GCP TPU (TRC) Helper Commands

Penrose Diffusion – TPU Research Cloud

Example:
```
  <TPU_ZONE> → us-central2-b
  <TPU_NAME> → penrose-test
```

## Auth & Project

Login
```
gcloud auth login
```

 Set project (hard-coded)
```
gcloud config set project penrose-diffusion
```

## Request 

### TRC-APPROVED TPU CREATE COMMANDS (SAFE TO USE)

```
Project: penrose-diffusion
TPU VM name: penrose-train
```

TPU v4-8 — ON-DEMAND (MOST RELIABLE)
```
gcloud compute tpus tpu-vm create penrose-train \
  --project penrose-diffusion \
  --zone us-central2-b \
  --accelerator-type v4-8 \
  --version tpu-ubuntu2204-base
```

TPU v4-8 — SPOT
```
gcloud compute tpus tpu-vm create penrose-train \
  --project penrose-diffusion \
  --zone us-central2-b \
  --accelerator-type v4-8 \
  --version tpu-ubuntu2204-base \
  --preemptible
```

TPU v5e-8 — SPOT (US)
```
gcloud compute tpus tpu-vm create penrose-train \
  --project penrose-diffusion \
  --zone us-central1-a \
  --accelerator-type v5litepod-8 \
  --version tpu-ubuntu2204-base \
  --preemptible
```

TPU v5e-8 — SPOT (EU)
```
gcloud compute tpus tpu-vm create penrose-train \
  --project penrose-diffusion \
  --zone europe-west4-b \
  --accelerator-type v5litepod-8 \
  --version tpu-ubuntu2204-base \
  --preemptible
```

TPU v6e-8 — SPOT (US)
```
gcloud compute tpus tpu-vm create penrose-train \
  --project penrose-diffusion \
  --zone us-east1-d \
  --accelerator-type v6e-8 \
  --version tpu-ubuntu2204-base \
  --preemptible
```

TPU v6e-8 — SPOT (EU)
```
gcloud compute tpus tpu-vm create penrose-train \
  --project penrose-diffusion \
  --zone europe-west4-a \
  --accelerator-type v6e-8 \
  --version tpu-ubuntu2204-base \
  --preemptible
```

## 3. TPU Discovery & Status

List TPU VMs in a zone
```
gcloud compute tpus tpu-vm list \
  --project penrose-diffusion \
  --zone <TPU_ZONE>
```

Describe TPU
```
gcloud compute tpus tpu-vm describe <TPU_NAME> \
  --project penrose-diffusion \
  --zone <TPU_ZONE>
```

## 4. SSH into TPU VM (canonical)

SSH
```
gcloud compute tpus tpu-vm ssh <TPU_NAME> \
  --project penrose-diffusion \
  --zone <TPU_ZONE>
```

Show SSH command (for VS Code / debugging)
```
gcloud compute tpus tpu-vm ssh <TPU_NAME> \
  --project penrose-diffusion \
  --zone <TPU_ZONE> \
  --dry-run
```

## 5. TPU Sanity Checks (inside VM)

TPU devices
```
ls /dev/accel*
```

XLA device check
```
python - << 'EOF'
import torch_xla.core.xla_model as xm
print(xm.xla_device())
EOF
```


## 6. Install PyTorch + torch_xla (inside VM)

```

pip install --upgrade pip

pip install torch torchvision torch_xla[tpu] \
  -f https://storage.googleapis.com/libtpu-releases/index.html
```



## 7. Git Workflow (inside VM)


Clone repo
```

git clone https://github.com/rakeshvar/penrose_diffusion
cd penrose_diffusion
```

Pull updates
```

git pull
```

If local edits exist
```
git stash
git pull
git stash pop
```

Hard reset
```
git fetch origin
git reset --hard origin/main
```



## 8. GCS Bucket (hard-coded)


Bucket used everywhere:
```
gs://penrose_diffusion
```


Create bucket (once)
```
gsutil mb -l us-central1 gs://penrose_diffusion
```

List contents
```
gsutil ls gs://penrose_diffusion/
```

Create placeholder "folders"
```
echo "" | gsutil cp - gs://penrose_diffusion/checkpoints/.keep
echo "" | gsutil cp - gs://penrose_diffusion/samples/.keep
```



## 9. Upload / Download Data


Upload dataset
```
gsutil cp data.npz gs://penrose_diffusion/datasets/data.npz
```

Download dataset
```
mkdir -p datasets
gsutil cp gs://penrose_diffusion/datasets/data.npz datasets/
```

Sync checkpoints
```
gsutil rsync -r checkpoints gs://penrose_diffusion/checkpoints
```

Sync samples
```
gsutil rsync -r samples gs://penrose_diffusion/samples
```



## 10. Training Runs


CPU sanity run
```
python train.py datasets/data.npz toy \
  -t batch_size=2 \
  -t epochs=1
```

TPU test run
```
python train.py datasets/data.npz toy \
  -t batch_size=2 \
  -t epochs=1 \
  -t lr=0.0001
```

Resume from checkpoint
```
python train.py checkpoints/latest.pt
```



## 11. tmux (fire-and-forget)


Start tmux
```
tmux new -s run1
```

Detach
```
Ctrl+B then D
```

List sessions
```
tmux ls
```

Reattach
```
tmux attach -t run1
```



## 12. Cleanup (IMPORTANT)


Delete TPU VM
```
gcloud compute tpus tpu-vm delete <TPU_NAME> \
  --project penrose-diffusion \
  --zone <TPU_ZONE>
```
