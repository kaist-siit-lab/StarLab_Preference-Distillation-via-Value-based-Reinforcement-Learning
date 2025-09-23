# Preference Distillation via Value-Based Reinforcement Learning

Official repository of **Preference Distillation via Value-Based Reinforcement Learning** (NeurIPS 2025 Poster).  
This repository provides the implementation of **TVKD (Teacher Value-Based Knowledge Distillation)**.

---

## 📌 Overview
TVKD training consists of three main stages:

1. **Train the Teacher Model**
2. **Supervised Fine-Tuning (SFT) for Student Initialization**
3. **Run TVKD**

---

## ⚙️ Environment Setup
1. Prepare an environment with **PyTorch** installed according to your GPU setup.  
2. Clone this repository and install the dependencies:
   ```bash
   git clone <your-repo-url>
   cd <your-repo>
   pip install -r requirements.txt
   ```
3. The implementation mainly depends on:
   - [trl](https://github.com/huggingface/trl)
   - [peft](https://github.com/huggingface/peft)
   - [transformers](https://github.com/huggingface/transformers)
   - [vllm](https://github.com/vllm-project/vllm)

   As long as your Hugging Face versions are aligned, it should run without issues.

**Recommended versions**:
- `torch==2.5.1`
- `trl==0.12.0`
- `peft==0.13.0`

---

## 🚀 Training Procedure

### 1. Train the Teacher Model
Prepare an initial teacher model (e.g., **LLaMA, Mistral**).  
Since TVKD assumes a **DPO-trained teacher**, you must first perform **DPO training**.  
Optionally, you may run **SFT on the teacher** before DPO to obtain a stronger baseline.

In our setup, we follow the paper and use **LLaMA-3.1-8B** as the teacher.  
- **SFT dataset:** `Deita-10k-V0`  
- **DPO dataset:** `DPO-MIX-7K`

#### (a) SFT Training for REF Teacher
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
ACCELERATE_LOG_LEVEL=info \
DS_SKIP_CUDA_CHECK=1 \
python -m accelerate.commands.launch \
  --config_file recipes/accelerate_config/deepspeed_zero3.yaml \
  scripts/run_sft.py \
  recipes/llama3.2-1b-deita-dpomix/teacher_sft.yaml
```

#### (b) DPO Training on the REF Teacher
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
ACCELERATE_LOG_LEVEL=info \
DS_SKIP_CUDA_CHECK=1 \
python -m accelerate.commands.launch \
  --config_file recipes/accelerate_config/deepspeed_zero3.yaml \
  scripts/run_distill_dpo.py \
  recipes/llama3.2-1b-deita-dpomix/teacher_dpo.yaml
```

> **Note:** You can adjust hyperparameters via the corresponding YAML files.

---

### 2. SFT for Student Initialization
The student model is initialized via **SFT** using the same dataset as the teacher.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
ACCELERATE_LOG_LEVEL=info \
DS_SKIP_CUDA_CHECK=1 \
python -m accelerate.commands.launch \
  --config_file recipes/accelerate_config/deepspeed_zero3.yaml \
  scripts/run_sft.py \
  recipes/llama3.2-1b-deita-dpomix/student_sft_init.yaml
```

---

### 3. Run TVKD
Finally, run TVKD training with:
```bash
./run/run_tvkd.sh
```

---

## 📄 Citation
If you use this repository or the TVKD method in your research, please cite our NeurIPS 2025 paper:

```bibtex
@inproceedings{tvkd2025,
  title     = {Preference Distillation via Value-Based Reinforcement Learning},
  author    = {...},
  booktitle = {NeurIPS},
  year      = {2025}
}
```
