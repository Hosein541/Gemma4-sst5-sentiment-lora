# Fine-Grained Sentiment Classification on SST-5 using Gemma & LoRA

An efficient instruction-tuning pipeline for **5-class fine-grained sentiment analysis** (Very Negative to Very Positive) on the Stanford Sentiment Treebank (SST-5) dataset, leveraging **Gemma (2B)**, **LoRA (PEFT)**, and accelerated via **Unsloth**.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/)
[![Unsloth](https://img.shields.io/badge/%F0%9F%A6%A5%20Unsloth-Fast%20Patching-green.svg)](https://github.com/unslothai/unsloth)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-orange.svg)](https://github.com/huggingface/peft)


---

## 📌 Overview

Fine-grained sentiment classification on the **SST-5 benchmark** is inherently challenging due to subtle stylistic nuances and boundaries between extreme and mild polarities (e.g., *Negative* vs. *Very Negative*).

Instead of traditional sequence classification heads, this project frames the task as a structured **Generative Instruction-Tuning Task**. Using **Unsloth** and **LoRA**, we fine-tune the model to output the exact class label with completion masking (`train_on_responses_only`), achieving high computational efficiency on a single GPU.

### Key Highlights:

* **Base LLM**: `unsloth/gemma-4-E2B` (Gemma architecture).
* **Parameter-Efficient Tuning**: Low-Rank Adaptation (LoRA) on all linear layers (`q`, `k`, `v`, `o`, `gate`, `up`, `down`).
* **Supervised Fine-Tuning (SFT)**: Utilized `TRL`'s `SFTTrainer` with response-only loss masking to focus gradient updates strictly on the label generation tokens.
* **Evaluation Accuracy**: Reached **60.40% overall accuracy** on the 5-class test set with **0% unmatched/hallucinated outputs**.

---

## 📊 Dataset & Formatting

The model was trained on the [SetFit/sst5](https://huggingface.co/datasets/SetFit/sst5) dataset:

* **Classes**:
* `0`: Very Negative
* `1`: Negative
* `2`: Neutral
* `3`: Positive
* `4`: Very Positive


* **Pre-processing**: Combined all dataset splits and downsampled to achieve balanced class distributions.
* **Splits**: 70% Train, 10% Validation, 20% Test.

### Prompt Template

```text
### Instruction:
Classify the sentiment of this movie review into one of the classes (0, 1, 2, 3, 4):
- 0: Very Negative
- 1: Negative
- 2: Neutral
- 3: Positive
- 4: Very Positive

Review:
{text}

### Response:
{label}

```

---

## ⚙️ Hyperparameters & Training Setup

| Parameter | Configuration | Details |
| --- | --- | --- |
| **Base Model** | `unsloth/gemma-4-E2B` | 2B Parameter Generative Model |
| **Max Sequence Length** | 512 | Optimized for review lengths |
| **PEFT Method** | LoRA ($r=16, \alpha=16$) | Target: Q, K, V, O, Gate, Up, Down |
| **LoRA Dropout** | 0.0 | Optimized for Unsloth |
| **Effective Batch Size** | 16 | Batch size per device = 4, Grad Accum = 4 |
| **Optimizer** | `adamw_8bit` | VRAM-saving 8-bit AdamW |
| **Learning Rate** | `2e-4` | Linear decay with 5 warmup steps |
| **Training Steps** | 180 steps | Validation and checkpoints every 30 steps |
| **Best Model Metric** | `eval_loss` | Restored lowest loss checkpoint |

---

## 📈 Results & Evaluation

### Training Dynamics

| Step | Training Loss | Validation Loss |
| --- | --- | --- |
| 30 | 0.6405 | 0.6084 |
| 60 | 0.3514 | 0.4834 |
| 90 | 0.4798 | 0.4663 |
| 120 | 0.4055 | 0.4700 |
| **150** | **0.3869** | **0.4621 (Best)** |
| 180 | 0.6053 | 0.4663 |

### Test Set Performance (250 Samples)

* **Overall Accuracy**: **60.40%**
* **Unmatched Predictions**: **0** (All test outputs strictly conformed to the `[0-4]` schema)

| Class | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| **0: Very Negative** | 0.58 | 0.72 | **0.64** | 50 |
| **1: Negative** | 0.47 | 0.46 | 0.47 | 54 |
| **2: Neutral** | 0.62 | 0.49 | 0.55 | 49 |
| **3: Positive** | 0.62 | 0.55 | 0.58 | 47 |
| **4: Very Positive** | 0.74 | 0.80 | **0.77** | 50 |
| **Macro Average** | **0.61** | **0.61** | **0.60** | 250 |
| **Weighted Average** | **0.60** | **0.60** | **0.60** | 250 |

### Confusion Matrix

---

## 🚀 Quickstart & Inference

### 1. Installation

Clone the repository and install requirements:

```bash
git clone [https://github.com/](https://github.com/)/.git
cd 
pip install -r requirements.txt

```

---

## 📂 Repository Structure

```text
├── assets/
│   └── confusion_matrix.png        # Evaluation confusion matrix plot
├── Gemma_SST5_LoRA.ipynb       # Complete training and evaluation notebook
├── requirements.txt                # Pinned dependencies
└── README.md                       # Documentation

```

---

## 📜 Acknowledgements

* [Google Gemma](https://huggingface.co/google) for the foundation model.
* [Unsloth AI](https://github.com/unslothai/unsloth) for low-overhead fine-tuning and fast patching.
* [TRL](https://github.com/huggingface/trl) for the Supervised Fine-Tuning (SFT) implementation.

```

```

---
