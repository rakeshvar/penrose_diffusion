
for z in \
  europe-west4-a \
  us-east1-d \
  us-central2-b \
  us-central1-a \
  europe-west4-b 
do
  echo "---- $z ----"
  gcloud compute tpus tpu-vm list --zone $z || true
done
