#! /bin/bash
# usage: bash scripts/general_domain/process_data.sh v1|v2
VERSION=${1:-v1}

export TF_CPP_MIN_LOG_LEVEL=3

PYTHONPATH=. python3 ./tools/process_data.py \
    --data-dir ./data/general_domain/ee_${VERSION}/ \
    --processed-data-dir ./processed_data/general_domain/ee_${VERSION} \
    --model-path Qwen/Qwen3-0.6B \
    --data-process-workers 4 \
    --max-prompt-length 460 \
    --dev-num 1000 \
    --model-type qwen
