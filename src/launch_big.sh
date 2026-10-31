#!/bin/bash
# Poll GPUs; when one has >= 9.3 GB free, start a worker for the 7-8B models there (once). Stops on logs/STOP_workers.
cd "$(dirname "$0")/.."
export NUMPY_MADVISE_HUGEPAGE=0 HF_HOME=/data/transformers HF_HUB_OFFLINE=1
while [ ! -f logs/STOP_workers ]; do
  read g f < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -n -r | head -1 | tr -d ,)
  if [ "$f" -ge 9300 ]; then
    echo "$(date +%T) launching big worker on GPU $g ($f MiB free)"
    CUDA_VISIBLE_DEVICES=$g .venv/bin/python src/worker.py logs/queue_big2.txt 9.0 > logs/worker_big_gpu$g.log 2>&1
    echo "$(date +%T) big worker exited rc=$?"
    exit 0
  fi
  sleep 3
done
