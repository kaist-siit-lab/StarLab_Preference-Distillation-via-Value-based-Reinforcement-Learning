Preference Distillation via Value-Based Reinforcement Learning
This repository provides an implementation of Teacher Value-Based Knowledge Distillation (TVKD), a method proposed in the paper Preference Distillation via Value-Based Reinforcement Learning (NeurIPS 2025 Poster).

To implement TVKD, you need to follow these three essential steps:

Train the Teacher Model

Supervised Fine-Tuning (SFT) for Student Initialization

Run TVKD

Environment Setup
First, make sure you have a Python environment with a torch installation that's compatible with your GPU setup.

After that, clone this repository and install the required dependencies:

Bash

git clone <repository_url>
pip install -r requirements.txt
This code primarily requires trl, peft, transformers, and vllm. As long as you match the Hugging Face versions, you should be able to run it without any major issues. The recommended versions are torch==2.5.1, trl==0.12.0, and peft==0.13.0.

Training Procedure
1. Train the Teacher Model 🧑‍🏫
You'll need an initial Teacher model (e.g., LLaMA, Mistral) to start with. Since we assume the Teacher is trained with DPO, you'll need to run the DPO training phase first. Optionally, SFTing the Teacher model beforehand is a great way to get an even better Teacher.

For this project, we use LLaMA 3.1-8B as the Teacher, as provided in the paper. We'll use the Deita-10k-V0 dataset for SFT and the DPO-MIX-7K dataset for DPO.

(a) SFT training to get the REF teacher
Use this command to run Supervised Fine-Tuning and get the initial reference teacher.

Bash

CUDA_VISIBLE_DEVICES=0,1,2,3 \
ACCELERATE_LOG_LEVEL=info \
DS_SKIP_CUDA_CHECK=1 \
python -m accelerate.commands.launch \
  --config_file recipes/accelerate_config/deepspeed_zero3.yaml \
  scripts/run_sft.py \
  recipes/llama3.2-1b-deita-dpomix/teacher_sft.yaml
(b) DPO training on the REF teacher to get the DPO teacher
Next, run DPO training on the SFT-trained reference teacher to create the DPO-aligned teacher.

Bash

CUDA_VISIBLE_DEVICES=0,1,2,3 \
ACCELERATE_LOG_LEVEL=info \
DS_SKIP_CUDA_CHECK=1 \
python -m accelerate.commands.launch \
  --config_file recipes/accelerate_config/deepspeed_zero3.yaml \
  scripts/run_distill_dpo.py \
  recipes/llama3.2-1b-deita-dpomix/teacher_dpo.yaml
You can customize your training settings by adjusting the corresponding YAML files.

2. SFT for Student Initialization 👨‍🎓
Next, train the Student model. The Student is initialized with SFT using the same dataset as the Teacher.

Bash

CUDA_VISIBLE_DEVICES=0,1,2,3 \
ACCELERATE_LOG_LEVEL=info \
DS_SKIP_CUDA_CHECK=1 \
python -m accelerate.commands.launch \
  --config_file recipes/accelerate_config/deepspeed_zero3.yaml \
  scripts/run_sft.py \
  recipes/llama3.2-1b-deita-dpomix/student_sft_init.yaml
3. Run TVKD 🚀
Finally, execute the main TVKD script to perform the knowledge distillation.

Bash

./run/run_tvkd.sh
