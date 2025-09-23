#!/bin/bash

set -e  
set -o pipefail  
trap 'echo -e "\n❌ [ERROR] Command failed: $BASH_COMMAND\n"' ERR


LOG_DIR="logs"
mkdir -p "$LOG_DIR"


log_and_run () {
  local NAME="$1"
  shift
  local LOG_FILE="$LOG_DIR/${NAME}.log"

  echo -e "\n🔹 [START] $NAME"
  echo "🔸 Logging to: $LOG_FILE"
  echo "------------------------------------------------"

  {
    echo ">>> [Start: $(date)]"
    echo ">>> [Command] $@"
    echo "------------------------------------------------"
    eval "$@"  
    echo ">>> [Success: $(date)]"
  } 2>&1 | tee "$LOG_FILE"

  echo -e "✅ [DONE] $NAME\n"
}

log_and_run "precompute_train_chosen" \
"CUDA_VISIBLE_DEVICES=0,1,2,3 python -m accelerate.commands.launch \
  --num_processes=4 \
  --main_process_port 29501 \
  utils/precompute_logits.py \
  --data argilla/dpo-mix-7k \
  --split train \
  --model ./model/dpo_teacher \
  --conversation-key chosen \
  --user-begin '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n' \
  --user-end '<|eot_id|>' \
  --assistant-begin '<|start_header_id|>assistant<|end_header_id|>\n\n' \
  --assistant-end '<|eot_id|>' \
  --save-to data/llama3.2-1b-deita-dpomix/dpomix7k-dpoteacher-chosen-logp-train \
  --pad-token-id 128001 \
  --max-tokens-per-batch 2048"
log_and_run "rm_train_chosen_temp" \
"rm data/llama3.2-1b-deita-dpomix/dpomix7k-dpoteacher-chosen-logp-train/results_rank_*.jsonl"


# Repeat similarly for:
#   - "chosen" / "rejected" + train/test
# train-rejected
log_and_run "precompute_train_rejected" \
"CUDA_VISIBLE_DEVICES=0,1,2,3 python -m accelerate.commands.launch \
  --num_processes=4 \
  --main_process_port 29501 \
  utils/precompute_logits.py \
  --data argilla/dpo-mix-7k \
  --split train \
  --model ./model/dpo_teacher \
  --conversation-key rejected \
  --user-begin '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n' \
  --user-end '<|eot_id|>' \
  --assistant-begin '<|start_header_id|>assistant<|end_header_id|>\n\n' \
  --assistant-end '<|eot_id|>' \
  --save-to data/llama3.2-1b-deita-dpomix/dpomix7k-dpoteacher-rejected-logp-train \
  --pad-token-id 128001 \
  --max-tokens-per-batch 2048"

log_and_run "rm_train_rejected_temp" \
"rm data/llama3.2-1b-deita-dpomix/dpomix7k-dpoteacher-rejected-logp-train/results_rank_*.jsonl"

# test-chosen
log_and_run "precompute_test_chosen" \
"CUDA_VISIBLE_DEVICES=0,1,2,3 python -m accelerate.commands.launch \
  --num_processes=4 \
  --main_process_port 29501 \
  utils/precompute_logits.py \
  --data argilla/dpo-mix-7k \
  --split test \
  --model ./model/dpo_teacher \
  --conversation-key chosen \
  --user-begin '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n' \
  --user-end '<|eot_id|>' \
  --assistant-begin '<|start_header_id|>assistant<|end_header_id|>\n\n' \
  --assistant-end '<|eot_id|>' \
  --save-to data/llama3.2-1b-deita-dpomix/dpomix7k-dpoteacher-chosen-logp-test \
  --pad-token-id 128001 \
  --max-tokens-per-batch 2048"

log_and_run "rm_test_chosen_temp" \
"rm data/llama3.2-1b-deita-dpomix/dpomix7k-dpoteacher-chosen-logp-test/results_rank_*.jsonl"

# test-rejected
log_and_run "precompute_test_rejected" \
"CUDA_VISIBLE_DEVICES=0,1,2,3 python -m accelerate.commands.launch \
  --num_processes=4 \
  --main_process_port 29501 \
  utils/precompute_logits.py \
  --data argilla/dpo-mix-7k \
  --split test \
  --model ./model/dpo_teacher \
  --conversation-key rejected \
  --user-begin '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n' \
  --user-end '<|eot_id|>' \
  --assistant-begin '<|start_header_id|>assistant<|end_header_id|>\n\n' \
  --assistant-end '<|eot_id|>' \
  --save-to data/llama3.2-1b-deita-dpomix/dpomix7k-dpoteacher-rejected-logp-test \
  --pad-token-id 128001 \
  --max-tokens-per-batch 2048"

log_and_run "rm_test_rejected_temp" \
"rm data/llama3.2-1b-deita-dpomix/dpomix7k-dpoteacher-rejected-logp-test/results_rank_*.jsonl"

log_and_run "merge" \
"python utils/merge_logits_dckd_dataset.py \
    --input-dataset-dict          argilla/dpo-mix-7k \
    --teacher-chosen-logp-train   data/llama3.2-1b-deita-dpomix/dpomix7k-dpoteacher-chosen-logp-train \
    --teacher-rejected-logp-train data/llama3.2-1b-deita-dpomix/dpomix7k-dpoteacher-rejected-logp-train \
    --teacher-chosen-logp-test    data/llama3.2-1b-deita-dpomix/dpomix7k-dpoteacher-chosen-logp-test \
    --teacher-rejected-logp-test  data/llama3.2-1b-deita-dpomix/dpomix7k-dpoteacher-rejected-logp-test \
    --save-to                     data/llama3.2-1b-deita-dpomix/dpomix7k-dckd"
