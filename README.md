# SABER: Switchable and Balanced Training for Efficient LLM Reasoning

This repository contains the training code for SABER, a reinforcement learning framework that enables LLMs to reason under user-controllable, token-budgeted modes.

**Paper**: *SABER: Switchable and Balanced Training for Efficient LLM Reasoning* (AAAI 2026)

## Overview

SABER addresses the *overthinking problem* in large reasoning models — where LLMs generate unnecessarily elaborate reasoning even for trivial inputs. It trains a single model that supports four discrete inference modes with different reasoning depths:

| Mode | Description | Use Case |
|------|-------------|----------|
| **NoThink** | Direct answer, no reasoning | Simple queries, low latency |
| **FastThink** | Budget ≤ 128 tokens | Easy problems |
| **CoreThink** | Budget ≤ 4,096 tokens | Medium difficulty |
| **DeepThink** | Unrestricted reasoning | Hard problems |

Key results (DeepSeek-R1-Distill-Qwen-1.5B base):
- **FastThink** cuts reasoning length by 72.7% while improving accuracy by 3.0%
- **CoreThink** reduces length by 67.9% with +4.7% accuracy
- **DeepThink** reduces length by 41.2% with +6.8% accuracy
- Generalizes across scales (1.5B → 7B) and domains (math → code → logic)

## Method

SABER's training pipeline consists of three stages:

1. **Budget Categorization**: Profile base model thinking token usage per example, assign to difficulty tiers (Easy/Medium/Hard)
2. **Curriculum-style Budget Downgrade**: Progressively assign tighter token budgets with stability mechanisms
3. **RL Training with Composite Reward**: Train with GRPO using four reward components:
   - `r_format`: Output format correctness (`<think>...</think>` structure)
   - `r_answer`: Answer correctness (exact match / code execution)
   - `r_length`: Penalty if thinking tokens exceed budget
   - `r_ratio`: Lower-bound constraint (0.2 × t_base ≤ t_gen ≤ 1.2 × t_base) to prevent reward hacking

## Installation

```bash
pip install -e .
```

**Requirements**: Python ≥ 3.10, PyTorch, vLLM (≥ 0.8.5), Ray (≥ 2.41.0), Transformers

## Training

### Quick Start (Single Node)

```bash
bash saber_train.sh
```

### Configuration

The training script uses Hydra-based configuration. Key parameters:

```bash
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/train.parquet \
    data.val_files=data/val.parquet \
    data.max_prompt_length=1024 \
    data.max_response_length=20480 \
    actor_rollout_ref.model.path=<model_path> \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    reward_model.reward_manager=prime \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=10
```

### Data Format

Training data should be a parquet file with columns:
- `prompt`: The input question/problem
- `data_source`: Task identifier (e.g., `math`, `code_contests`, `gsm8k`)
- `reward_model.ground_truth`: Expected answer for reward computation
- `extra_info`: (optional) Additional metadata including budget tier

### Reward Functions

SABER uses the PRIME reward manager with customizable scoring functions:
- **math**: Math answer extraction from `\boxed{}` and verification (supports math_verify)
- **sandbox**: Code execution in sandbox environment against test cases
- **saber**: Token budget length scoring (thinking length penalty + lower-bound ratio constraint)

## Project Structure

```
saber_rl/
├── saber_train.sh              # Training launch script
├── setup.py                    # Package installation
└── verl/                       # Core verl framework
    ├── trainer/                # Training orchestration
    │   ├── main_ppo.py        # PPO/GRPO entry point
    │   ├── ppo/               # Core RL algorithms
    │   └── config/            # Hydra YAML configs
    ├── workers/                # Distributed workers
    │   ├── fsdp_workers.py    # FSDP actor/critic/ref
    │   ├── rollout/           # vLLM/SGLang rollout
    │   └── reward_manager/    # Reward computation
    ├── utils/                  # Utilities
    │   ├── reward_score/      # Reward scoring (math, sandbox, saber)
    │   ├── dataset/           # Data loading
    │   └── checkpoint/        # Checkpoint management
    └── models/                 # Model definitions (Qwen2, LLaMA)
```

## Citation

```bibtex
@inproceedings{zhao2026saber,
  title={Saber: Switchable and balanced training for efficient llm reasoning},
  author={Zhao, Kai and Zhao, Yanjun and Song, Jiaming and He, Shien and Zhang, Lusheng and Zhang, Qiang and Li, Tianjiao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={41},
  pages={34950--34958},
  year={2026}
}
```

## License

Apache License 2.0
