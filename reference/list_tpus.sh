
for z in \
  europe-west4-a \
  us-east1-d \
  us-central2-b \
  us-central1-a \
  europe-west4-b
do
  echo "---- $z ----"
  gcloud compute tpus tpu-vm list --zone $z || true
  gcloud compute tpus tpu-vm describe penrose-train --zone=$z \
    --format="value(state,networkEndpoints[0].accessConfig.externalIp)" 2>/dev/null | \
    awk '$1=="READY" {print "External IP: " $2}'
done
