
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m accelerate.commands.launch \
  --num_processes=4 \
  --main_process_port 29501 \
  utils/precompute_logits.py \
  --data argilla/dpo-mix-7k \
  --split train \
  --model home/minchan.kwon/ADPA/ADPA-OpenSource/data/llama3.2-1b-deita-dpomix/dpo_teacher \
  --conversation-key rejected \
  --user-begin "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" \
  --user-end "<|eot_id|>" \
  --assistant-begin "<|start_header_id|>assistant<|end_header_id|>\n\n" \
  --assistant-end "<|eot_id|>" \
  --save-to data/llama3.2-1b-deita-dpomix/dpomix7k-dpoteacher-rejected-logp-train \
  --pad-token-id 128001 \
  --max-tokens-per-batch 2048
rm data/llama3.2-1b-deita-dpomix/dpomix7k-dpoteacher-rejected-logp-train/results_rank_*.jsonl

# test-chosen
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m accelerate.commands.launch \
  --num_processes=4 \
  --main_process_port 29501 \
  utils/precompute_logits.py \
  --data argilla/dpo-mix-7k \
  --split test \
  --model home/minchan.kwon/ADPA/ADPA-OpenSource/data/llama3.2-1b-deita-dpomix/dpo_teacher \
  --conversation-key chosen \
  --user-begin "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" \
  --user-end "<|eot_id|>" \
  --assistant-begin "<|start_header_id|>assistant<|end_header_id|>\n\n" \
  --assistant-end "<|eot_id|>" \
  --save-to data/llama3.2-1b-deita-dpomix/dpomix7k-dpoteacher-chosen-logp-test \
  --pad-token-id 128001 \
  --max-tokens-per-batch 2048
rm data/llama3.2-1b-deita-dpomix/dpomix7k-dpoteacher-chosen-logp-test/results_rank_*.jsonl

# test-rejected
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m accelerate.commands.launch \
  --num_processes=4 \
  --main_process_port 29501 \
  utils/precompute_logits.py \
  --data argilla/dpo-mix-7k \
  --split test \
  --model home/minchan.kwon/ADPA/ADPA-OpenSource/data/llama3.2-1b-deita-dpomix/dpo_teacher \
  --conversation-key rejected \
  --user-begin "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" \
  --user-end "<|eot_id|>" \
  --assistant-begin "<|start_header_id|>assistant<|end_header_id|>\n\n" \
  --assistant-end "<|eot_id|>" \
  --save-to data/llama3.2-1b-deita-dpomix/dpomix7k-dpoteacher-rejected-logp-test \
  --pad-token-id 128001 \
  --max-tokens-per-batch 2048
rm data/llama3.2-1b-deita-dpomix/dpomix7k-dpoteacher-rejected-logp-test/results_rank_*.jsonl