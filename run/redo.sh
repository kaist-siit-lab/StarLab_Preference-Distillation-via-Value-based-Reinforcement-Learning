#!/bin/bash

# 반복 횟수 설정
num_cycles=5

# 초기 student model 경로
student_model_path="./model/student_sft_init"

# reward model 고정
reward_model="sfairXC/FsfairX-LLaMA3-RM-v0.1"

# output 파일 및 모델 저장 폴더
mkdir -p train_offline
mkdir -p models

for ((i=1; i<=num_cycles; i++))
do
  echo "================== Cycle $i: Offline Generation =================="

  output_json="train_offline/epoch_${i}.jsonl"
  
  python scripts/offline_generation.py \
    --dataset_name argilla/dpo-mix-7k \
    --dataset_split train \
    --max_samples 5000 \
    --student_model_name_or_path ${student_model_path} \
    --reward_model_name_or_path ${reward_model} \
    --output_file ${output_json} \
    --num_return_sequences 2 \
    --num_beams 4 \
    --do_sample \
    --temperature 1.2 \
    --top_p 0.9 \
    --max_new_tokens 500 \
    --batch_size 16 \

  echo "================== Cycle $i: DPO Training =================="

  output_model_dir="models/model_epoch_${i}"

  python scripts/dpo_training.py \
    --train_file ${output_json} \
    --eval_file test_offline.jsonl \
    --student_model_name_or_path ${student_model_path} \
    --reward_model_name_or_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
    --output_dir ${output_model_dir} \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5

  # 다음 cycle에서 사용할 student 모델 경로 업데이트
  student_model_path="${output_model_dir}"
done