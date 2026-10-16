#!/bin/bash
# 350M pretraining on a RunPod pod (PyTorch template, 80 GB GPU). Needs an HF token (HF_TOKEN or hf auth login).
# Usage: bash pod_pretrain.sh
set -e
REPO=RobinHU73/vintage_llm
DIR=/workspace/vintage_llm   # container disk: wiped if the pod stops, so HF holds the checkpoints

pip install -q -U huggingface_hub tokenizers python-dotenv
hf download $REPO --repo-type dataset --local-dir $DIR \
    --include "model/*.py" --include "bpe/*" --include "data/processed/clean/*.bin" \
    --include "model/out/pre350_last.pt"   # one pattern per --include; a fresh pod resumes from pre350_last.pt

cd $DIR/model
mkdir -p out
# Every 90 min, push the log and resume checkpoint to HF. Each upload adds 4.3 GB to the repo's history
# (which counts against storage), so: one file, sparse uploads, and errors are logged, not hidden.
(while sleep 5400; do
    hf upload $REPO out/pre350_log.csv model/out/pre350_log.csv --repo-type dataset -q || echo "UPLOAD FAILED: log"
    hf upload $REPO out/pre350_last.pt model/out/pre350_last.pt --repo-type dataset -q || echo "UPLOAD FAILED: last"
done) 2>&1 | tee -a out/upload.log &

NGPU=$(nvidia-smi -L | wc -l)
if [ "$NGPU" -gt 1 ]; then  # memmap, not --in-memory: 8 ranks x 17 GB would fill RAM
    torchrun --standalone --nproc_per_node $NGPU pretrain.py --compile 2>&1 | tee -a out/pre350_train.log
else
    python -u pretrain.py --compile --in-memory 2>&1 | tee -a out/pre350_train.log
fi
hf upload $REPO out model/out --repo-type dataset --include "pre350_*"
echo "ALL DONE"
