# SABER: Switchable and Balanced Training for Efficient LLM Reasoning

[![Paper](https://img.shields.io/badge/AAAI%202026-SABER-b31b1b.svg)](https://arxiv.org/abs/2508.10026)
[![Code](https://img.shields.io/badge/GitHub-saber__rl-blue.svg)](https://github.com/bilibili/saber_rl)

This repository contains the reinforcement learning training code for **SABER**, as described in:

> **SABER: Switchable and Balanced Training for Efficient LLM Reasoning**
>
> Kai Zhao\*, Yanjun Zhao\*, Jiaming Song, Shien He, Lusheng Zhang, Qiang Zhang, Tianjiao Li
>
> AAAI 2026

## Overview

SABER addresses the *overthinking problem* in large reasoning models by training a single model that supports four discrete inference modes with user-controllable reasoning depth:

| Mode | Budget | Use Case |
|------|--------|----------|
| **NoThink** | 0 tokens | Simple queries, low latency |
| **FastThink** | ≤ 128 tokens | Easy problems |
| **CoreThink** | ≤ 4,096 tokens | Medium difficulty |
| **DeepThink** | Unrestricted | Hard problems |

The RL stage uses **GRPO** (Group Relative Policy Optimization) with a composite reward to align the model's reasoning length with task-specific budgets:

```
r = r_format + r_answer + r_length + r_ratio
```

- **r_format**: Ensures `<think>...</think>` structure compliance (0 / -1)
- **r_answer**: Answer correctness via exact match or code execution (1 / 0)
- **r_length**: Penalty (-0.4) if thinking tokens exceed budget
- **r_ratio**: Lower-bound constraint (0.2 × t_base ≤ t_gen ≤ 1.2 × t_base) to prevent reward hacking

## Installation

```bash
pip install -e .
pip install vllm  # inference engine (or sglang)
```

## Training

```bash
bash saber_train.sh
```

Key parameters can be configured in the script. For multi-node training, set `trainer.nnodes=N`.

## Reward Function

The core reward implementation is in `verl/utils/reward_score/saber.py`.

| Component | Description |
|-----------|-------------|
| `r_format` | 0 if `<think>...</think>` structure is correct, -1 otherwise |
| `r_answer` | 1.0 if answer matches ground truth, 0.0 otherwise |
| `r_length` | -0.4 if thinking tokens exceed budget, 0 otherwise |
| `r_ratio` | -0.4 if generated length outside [0.2, 1.2] × base length |

## Data Format

Training data should be in parquet format with:
- `prompt`: Input question/problem
- `data_source`: Task identifier (e.g., `math`, `sandbox`)
- `reward_model.ground_truth`: Expected answer for reward computation
- `extra_info.token_upper`: Token budget for the thinking mode

## Project Structure

```
verl/
├── trainer/                    # GRPO trainer (Ray distributed)
├── workers/
│   └── reward_manager/         # Reward orchestration
├── utils/reward_score/
│   ├── saber.py                # Token budget length reward
│   ├── math.py                 # Math answer verification
│   └── sandbox.py              # Code execution reward
saber_train.sh                  # Training launch script
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

## Acknowledgements

Built on [verl](https://github.com/volcengine/verl) (Volcano Engine Reinforcement Learning for LLMs). Licensed under Apache-2.0.
