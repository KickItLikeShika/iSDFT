# Self-Distillation Fine-Tuning

This is TRL-based code for reproducing the On-Policy Self-Distillation algorithm from the paper "Self-Distillation Enables Continual Learning" - [https://arxiv.org/abs/2601.19897](https://arxiv.org/abs/2601.19897).

 All experiments can be run with a single H200 GPU. Other setups may require refactoring and/or changing model sizes.

### Updates

04/07/26: after some investigation, we've found that all the results in our paper were produced using on-policy sampling, but per-token forward KL loss (similar to the [GKD paper](https://arxiv.org/abs/2306.13649)). Therefore, this is the default argument in this repo, and we will update the arXiv version soon with clarification.

03/12/26: added the science dataset and evaluation pipeline, regenerated the tool-use dataset, and added an updated tool-use evaluation file. I'll upload the Medical and Wiki datasets soon.

## Abstract
Continual learning, enabling models to acquire new skills and knowledge without degrading existing capabilities, remains a fundamental challenge for foundation models. While on-policy reinforcement learning can reduce forgetting, it requires explicit reward functions that are often unavailable. Learning from expert demonstrations, the primary alternative, is dominated by supervised fine-tuning (SFT), which is inherently off-policy. We introduce On-Policy **Self-Distillation Fine-Tuning (SDFT)**, a simple method that enables on-policy learning directly from demonstrations. SDFT leverages in-context learning by using a demonstration-conditioned model as its own teacher, generating on-policy training signals that preserve prior capabilities while acquiring new skills. Across skill learning and knowledge acquisition tasks, SDFT consistently outperforms SFT, achieving higher new-task accuracy while substantially reducing catastrophic forgetting. In sequential learning experiments, SDFT enables a single model to accumulate multiple skills over time without performance regression, establishing on-policy distillation as a practical path to continual learning from demonstrations.


##  Setup

### 1. Clone the repository

```bash
git clone https://github.com/Continual-Intelligence/Self-Distillation.git
cd Self-Distillation
```

### 2. Set up a virtual environment

Using **conda**:

```bash
conda create -n distillation python=3.12
conda activate distillation
```

Using **venv**:

```bash
python3.12 -m venv distillation
source distillation/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Usage

#### Tooluse

Training:

```bash
python main.py \
  --dataset_name tooluse \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --output_dir <output_path> \
  --learning_rate 5e-5 \
  --num_train_epochs 2
```

Evaluation:

```bash
python eval_tooluse_simple.py \
  --model_path <path_to_trained_model> \
  --output_dir <output_path>
```

#### Science

Training:

```bash
python main.py \
  --dataset_name science \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --output_dir <output_path> \
  --learning_rate 5e-5 \
  --num_train_epochs 2
```

Evaluation:

```bash
python eval_science.py \
  --model_path <path_to_trained_model> \
  --output_dir <output_path>
```

### 5. Forgetting Evaluation

To produce the forgetting metrics in the paper we use the [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) by Eleuther AI.

To reproduce the results please install the specific commit we have used:
```bash
pip install git+https://github.com/EleutherAI/lm-evaluation-harness@03c44adc0586f88bb343a74da1a1c602103536dd
```

and run the following command:

```bash
lm_eval --model hf --model_args pretrained=<path_to_your_model> --output_path <output_dir> --confirm_run_unsafe_code --tasks hellaswag,mmlu,truthfulqa,winogrande,humaneval,ifeval
```
