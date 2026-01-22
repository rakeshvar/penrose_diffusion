#!/usr/bin/env bash
set -euo pipefail

TPU_NAME=${1:-penrose-train}
PROJECT=${2:-penrose-diffusion}

echo "Creating TPU VM: $TPU_NAME"
echo "Project: $PROJECT"
echo "======================================"

try_create() {
  ZONE=$1
  ACCEL=$2
  PREEMPTIBLE=$3
  VERSION=$4
  LABEL=$5

  echo ""
  echo ">>> Trying $LABEL"
  echo "    zone=$ZONE accelerator=$ACCEL preemptible=$PREEMPTIBLE version=$VERSION"

  CMD=(
    gcloud compute tpus tpu-vm create "$TPU_NAME"
    --project="$PROJECT"
    --zone="$ZONE"
    --accelerator-type="$ACCEL"
    --version="$VERSION"
  )

  if [[ "$PREEMPTIBLE" == "true" ]]; then
    CMD+=(--preemptible)
  fi

  if "${CMD[@]}"; then
    echo ""
    echo "✅ SUCCESS: $LABEL"
    echo "TPU VM '$TPU_NAME' is ready in zone $ZONE"
    echo ""
    echo "To connect:"
    echo "  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE"
    echo -e "\a"
    sleep 1
    echo -e "\a"
    sleep 1
    echo -e "\a"
    exit 0
  fi
}

# -------------------------------
# Try in recommended order
# CRITICAL: v6e requires v2-alpha-tpuv6e runtime
# v5e and v4 can use tpu-ubuntu2204-base or tpu-vm-v4-base
# -------------------------------

# 2. v6e spot (newer, requires specific runtime)
try_create europe-west4-a v6e-8 true "v2-alpha-tpuv6e" "TPU v6e SPOT (europe-west4-a)"
try_create us-east1-d     v6e-8 true "v2-alpha-tpuv6e" "TPU v6e SPOT (us-east1-d)"

# 1. v5e spot (often fastest, well-supported)
try_create europe-west4-b v5litepod-8 true "tpu-ubuntu2204-base" "TPU v5e SPOT (europe-west4-b)"
try_create us-central1-a  v5litepod-8 true "tpu-ubuntu2204-base" "TPU v5e SPOT (us-central1-a)"
# v2-alpha-tpuv5-lite

# 3. v4 spot (most reliable spot option)
try_create us-central2-b v4-8 true "tpu-ubuntu2204-base" "TPU v4 SPOT (us-central2-b)"

# 4. v4 on-demand (most reliable, costs more)
try_create us-central2-b v4-8 false "tpu-ubuntu2204-base" "TPU v4 ON-DEMAND (us-central2-b)"
# tpu-ubuntu2204-base

echo ""
echo "❌ All TPU creation attempts failed."
echo "This is normal under TRC during peak usage."
echo "Try again later or remove --preemptible contention."
exit 1

# -------------------------------
#   64 spot Cloud TPU v5e chips in zone europe-west4-b
#   64 spot Cloud TPU v5e chips in zone us-central1-a
#   64 spot Cloud TPU v6e chips in zone europe-west4-a
#   64 spot Cloud TPU v6e chips in zone us-east1-d
#   32 spot Cloud TPU v4 chips in zone us-central2-b
#   32 on-demand Cloud TPU v4 chips in zone us-central2-b

# gcloud compute tpus accelerator-types list --zone=YOUR_ZONE
# gcloud compute tpus versions list --zone=YOUR_ZONE