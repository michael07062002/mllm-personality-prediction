# MLLM Hidden-State Probing for Video Regression

This repository contains a pipeline for extracting hidden states from frozen multimodal language models and training a lightweight downstream regressor on top of them.

The main idea:

```text
video -> frozen MLLM -> hidden states -> top-3 layer selection -> downstream regression
```

The MLLM is not fine-tuned. Only the downstream regression model is trained.

---

## Supported Tasks

### First Impressions V2

Video-level Big Five personality regression.

Targets:

```text
O, C, E, A, N
```

### Aff-Wild2 VA

Video-level affect regression.

Targets:

```text
valence, arousal
```

---

## Project Structure

```text
.
├── config/
│   ├── dlsp.yaml
│   ├── features.yaml
│   └── train.yaml
│
├── data/
│   ├── FirstImpressionsV2/
│   └── AffWild2_CVPR-26/
│
├── experiments/
├── outputs/
│
├── run_dlsp.py
├── run_features.py
├── train.py
│
└── src/
    ├── extraction/
    ├── features/
    └── training/
```

---

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset Placement

All datasets should be placed inside the local `data/` directory.

Do not put absolute local paths into configs.

### First Impressions V2

Expected structure:

```text
data/
└── FirstImpressionsV2/
    ├── annotations/
    │   ├── annotation_training.pkl
    │   └── annotation_test.pkl
    │
    └── data/
        ├── train/
        └── test/
```

### Aff-Wild2 VA

Expected structure:

```text
data/
└── AffWild2_CVPR-26/
    ├── videos/
    │   └── ...
    │
    └── Annotations/
        └── VA_Estimation_Challenge/
            ├── Train_Set/
            └── Validation_Set/
```

---

## Supported Datasets

Use these names in configs and commands:

```text
first_impressions
affwild2_va
```

---

## Supported Models

Use these names in configs and commands:

```text
qwen25_vl_3b
qwen3_vl_2b
internvl3_2b
smolvlm_256m
```

Model mapping:

| Short name | Hugging Face model |
|---|---|
| `qwen25_vl_3b` | `Qwen/Qwen2.5-VL-3B-Instruct` |
| `qwen3_vl_2b` | `Qwen/Qwen3-VL-2B-Instruct` |
| `internvl3_2b` | `OpenGVLab/InternVL3-2B-hf` |
| `smolvlm_256m` | `HuggingFaceTB/SmolVLM-256M-Instruct` |

---

## Pipeline

The repository has three main stages:

```text
run_dlsp.py      -> selects informative hidden layers
run_features.py  -> extracts selected hidden-state features
train.py         -> trains the downstream regression model
```

---

## 1. DLSP Layer Selection

DLSP selects informative hidden layers using statistics computed between low/high target groups.

Config file:

```text
config/dlsp.yaml
```

Example:

```yaml
dataset: affwild2_va
model: qwen25_vl_3b
device: auto
```

Run:

```bash
python run_dlsp.py
```

or:

```bash
python run_dlsp.py --dataset affwild2_va --model qwen25_vl_3b
```

Output:

```text
experiments/<dataset_experiment>/<model>/analysis/top_layers.json
```

Example:

```json
{
  "best_layer": 17,
  "top_layers": [17, 19, 34]
}
```

The usual setting is to use the top-3 selected layers.

---

## 2. Feature Extraction

Feature extraction saves segment-level hidden states from selected layers.

Config file:

```text
config/features.yaml
```

Recommended config:

```yaml
dataset: affwild2_va
model: qwen25_vl_3b
device: auto
layers: auto
```

`layers: auto` loads layers from:

```text
experiments/<dataset_experiment>/<model>/analysis/top_layers.json
```

Manual layer selection is also possible:

```yaml
dataset: first_impressions
model: qwen25_vl_3b
device: auto
layers: [layer_1, layer_2, layer_3]
```

Run:

```bash
python run_features.py
```

or:

```bash
python run_features.py --dataset first_impressions --model qwen25_vl_3b --layers layer_1,layer_2,layer_3
```

Output:

```text
experiments/<dataset_experiment>/<model>/layer_features/
```

Example:

```text
experiments/first_impressions_dlsp/qwen25_vl_3b/layer_features/
├── train/
│   ├── index.csv
│   ├── meta.json
│   ├── layer_18/
│   ├── layer_19/
│   └── layer_21/
└── test/
    ├── index.csv
    ├── meta.json
    ├── layer_18/
    ├── layer_19/
    └── layer_21/
```

---

## 3. Downstream Training

The downstream model is trained on extracted hidden-state features.

It supports:

- single-layer probing
- top-3 layer fusion
- AGF layer fusion
- Mamba-like temporal backbone
- regression head

Config file:

```text
config/train.yaml
```

Simple example:

```yaml
device: auto
compile: false

data:
  model_name: affwild2_va/qwen25_vl_3b
  layers: [layer_1, layer_2, layer_3]

model:
  d_model: 512
  d_state: 64
  d_conv: 4
  expand: 2
  num_mamba_blocks: 4
  mlp_ratio: 2.0
  dropout: 0.2
  agf_hidden_dim: 512

train:
  batch_size: 8
  num_workers: 0
  num_epochs: 100
  lr: 0.0001
  weight_decay: 0.001
  max_grad_norm: 0.5
  save_every_epoch_predictions: false

scheduler:
  use: true
  mode: max
  factor: 0.5
  patience: 5

outputs:
  root: outputs
```

Run:

```bash
python train.py
```

Training outputs are saved to:

```text
outputs/<dataset>/<model>/<run_name>/
```

Example:

```text
outputs/affwild2_va/qwen25_vl_3b/fusion_<layer_1>-<layer_2>-<layer_3>/
```

---

## Example Runs

### Run the full pipeline

```bash
python run_dlsp.py --dataset first_impressions --model qwen25_vl_3b
python run_features.py --dataset first_impressions --model qwen25_vl_3b
python train.py
```

### Use another dataset or model

```bash
python run_dlsp.py --dataset affwild2_va --model qwen25_vl_3b
python run_features.py --dataset affwild2_va --model qwen25_vl_3b
python train.py
```

After DLSP, use the top-3 layers from:

```text
experiments/<dataset_experiment>/<model>/analysis/top_layers.json
```

Put these three layers into `config/train.yaml`:

```yaml
data:
  model_name: first_impressions/qwen25_vl_3b
  layers: [layer_1, layer_2, layer_3]
```

---

## Generated Files

DLSP analysis:

```text
experiments/<dataset_experiment>/<model>/analysis/top_layers.json
```

Extracted features:

```text
experiments/<dataset_experiment>/<model>/layer_features/
```

Training results:

```text
outputs/<dataset>/<model>/<run_name>/
```

---

## What Should Not Be Committed

The following directories are generated locally and should not be committed:

```text
data/
experiments/
outputs/
```

They contain datasets, extracted features, logs, checkpoints, predictions, and experiment artifacts.

---

## Main Notes

- The MLLM is frozen.
- Hidden states are extracted from selected layers.
- DLSP is used to select informative layers.
- The usual setting is top-3 layer fusion.
- The downstream model is trained separately.
- First Impressions uses Big Five targets.
- Aff-Wild2 uses valence/arousal targets.
