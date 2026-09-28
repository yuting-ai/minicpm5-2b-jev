# MiniCPM5-2B-Jev

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Base Model](https://img.shields.io/badge/Base_Model-openbmb%2FMiniCPM5--2B-green)](https://huggingface.co/openbmb/MiniCPM5-2B)
[![Contract](https://img.shields.io/badge/API-%2Fv1%2Fsystemone-orange)](#43-http-server-v1systemone)

**`MiniCPM5-2B-Jev`** is an open-source, single-pass, calibrated **System 1 Decision Model** built on [`openbmb/MiniCPM5-2B`](https://huggingface.co/openbmb/MiniCPM5-2B). Given a shared context document (`state`) and a dictionary of typed decision questions (`choice`, `noul`, `score`), it returns a calibrated probability distribution over the options for every question in **one forward pass** — without autoregressive text generation.

It natively serves the `/v1/systemone` structured decision contract and ranks **#1 among all $\le 2\text{B}$ parameter open-weight System 1 / Jev models on `JevBench` (78.79%)**, outperforming `decider-2b v11` (76.2%), `Kev-4B` (75.8%), `system-one Qwen3-8B` (71.9%), and `Bespoke Nimble 9B` (67.5%), while achieving **86.41% accuracy and 0.0307 ECE** on the 10-source `decision-v7` benchmark.

---

## 1. Horizontal Benchmark Summary

### 1.1 JevBench Public Leaderboard Comparison (195 Groups / 231 Decisions)

Evaluated on the official [`fstandhartinger/jevbench`](https://github.com/fstandhartinger/jevbench) benchmark (`Easy`: 48, `Standard/Original`: 72, `Hard`: 111):

| Model | Base Model | Params | Easy (48) | Standard (72) | Hard (111) | Overall (231) | Open Weights |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Jev 1.13.0 *(TypeSafe Hosted)* | Proprietary | Closed | 100.0% | 98.6% | 73.0% | 86.6% | No |
| decider-35b-a3b v1 | Qwen3.5-35B-A3B (MoE) | 35B (3B act.) | 100.0% | 97.2% | 67.6% | 83.5% | Yes |
| decider-4b v2.1 | Qwen3.5-4B | 4.2B | 100.0% | 98.6% | 64.9% | 82.7% | Yes |
| OpenJev *(Hosted)* | DiffusionGemma 26B-A4B | 26B (4B act.) | 100.0% | 97.2% | 64.0% | 81.8% | No |
| SemIf | Qwen3.5-4B | 4.2B | 100.0% | 98.6% | 61.3% | 81.0% | Yes |
| ⭐ **MiniCPM5-2B-Jev (Ours)** | **openbmb/MiniCPM5-2B** | **2.0B** | **97.9%** | **94.4%** | **60.4%** | **78.8%** | **Yes (Apache-2.0)** |
| decider-2b v11 | Qwen3.5-2B | 1.9B | 100.0% | 88.9% | 57.7% | 76.2% | Yes |
| Kev-4B (r10) | Qwen3.5-4B-Base | 4.2B | — | — | 54.1% | 75.8% | Yes |
| open-alternative-jev | Qwen3.5-4B | 4.2B | 100.0% | 83.3% | 56.8% | 74.0% | Yes |
| system-one-open | Gemma 4 E2B | ~2B | 100.0% | 93.1% | 48.6% | 73.2% | Yes |
| system-one | Qwen3-8B | 8.2B | 100.0% | 88.9% | 48.6% | 71.9% | Yes |
| decider-2b v10 | Qwen3.5-2B | 1.9B | 100.0% | 88.9% | 45.9% | 70.6% | Yes |
| Bespoke Nimble 9B | Bespoke-Nimble-9B | 9.0B | 100.0% | 93.1% | 36.9% | 67.5% | No |
| Kev-0.8B (r15) | Qwen3.5-0.8B-Base | 0.8B | — | — | 36.0% | 63.6% | Yes |
| open-jev-deberta-v3-large | DeBERTa-v3-Large | 0.4B | 100.0% | 43.1% | 37.8% | 52.4% | Yes |

### 1.2 `decision-v7` 10-Source Typed Decision Benchmark

| Model | Base Model | Params | Accuracy ($\uparrow$) | Brier Score ($\downarrow$) | ECE ($\downarrow$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Kev-9B *(T=2.30)* | Qwen3.5-9B-Base | 9.0B | 87.2% | — | 0.042 |
| Kev-27B *(T=1.38)* | Qwen3.8-27B | 27.0B | 87.0% | — | — |
| ⭐ **MiniCPM5-2B-Jev (Ours)** | **openbmb/MiniCPM5-2B** | **2.0B** | **86.41%** | **0.2143** | **0.0307** |
| Kev-4B *(T=1.89)* | Qwen3.5-4B-Base | 4.2B | 85.90% | 0.2560 | 0.0560 |
| Jev *(TypeSafe Hosted)* | Proprietary | Closed | 84.50% | — | — |
| Kev-0.8B *(T=1.81)* | Qwen3.5-0.8B-Base | 0.8B | 77.10% | 0.3690 | 0.0880 |
| MiniCPM5-2B (Untuned Base, Step 0) | openbmb/MiniCPM5-2B | 2.0B | 62.50% | 0.6925 | 0.2715 |

### 1.3 All 7 Benchmark Suites Summary (`MiniCPM5-2B-Jev`)

| Benchmark Suite | States / Questions | Stage 1 (Top-20 LoRA) | **MiniCPM5-2B-Jev (Ours)** | Choice Acc | Noul Acc | Score Acc | Brier ($\downarrow$) | ECE ($\downarrow$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`decision-v7` (10-Source Stratified)** | 300 / 390 | 71.03% | **86.41%** | 87.10% | 94.97% | 67.11% | 0.2143 | **0.0307** |
| **`jabr v1` (8 Tasks / 78 Cases)** | 78 / 78 | 70.51% | **87.18%** | 96.00% | 76.92% | 88.89% | 0.1771 | **0.0277** |
| **`jabr v2` (49 Tasks / 869 Cases)** | 869 / 869 | 69.51% | **79.75%** | 85.16% | 79.49% | 69.95% | 0.2919 | **0.0663** |
| **`JevBench` (Overall Public Suite)** | 231 / 231 | 60.17% | **78.79%** | 78.42% | 81.08% | 72.22% | 0.3157 | **0.0699** |
| **`hard-v1` (Held-Out Template 4)** | 350 / 543 | 48.43% | **60.41%** | 58.35% | 70.43% | 40.00% | 0.5419 | **0.0603** |
| **`DecisionBench Medium`** | 80 / 293 | 49.15% | **60.07%** | 70.93% | 80.52% | 34.48% | 0.6596 | 0.2647 |
| **`DecisionBench Hard`** | 80 / 293 | 46.76% | **44.71%** | 53.75% | 70.79% | 23.33% | 0.8768 | 0.3711 |

---

## 2. Architecture & Technical Design

1. **Native `lm_head` Letter Readout (`LetterReadoutHead`)**:
   - Extracts the 255 single-token option letter embeddings (`A..Z` for `0..25`, `AA..` for `26..254`, shape `[255, 2304]`) directly from `MiniCPM5-2B`'s pretrained `lm_head.weight`.
   - Formats each question branch as `Question: [{type}] {instructions}\n(A) {opt_0}\n(B) {opt_1}\n...\nAnswer: (` and reads out the last-token hidden state $h_{\text{last}}$ at `(`.
   - Applies **Class-Balanced Marginal Prior Debiasing** (`noul_bias`, `choice_k_bias`, `score_k_bias`) and **Per-Type Temperature Scaling** (`choice: 1.1314`, `noul: 1.2338`, `score: 0.4213`), stored directly inside `head.pt`.
2. **Shared-Prefix KV-Cache Multi-Question Branching**:
   - When multiple questions share a long `state` ($\ge 256$ tokens), the model encodes `State: {state}\n\n` once with `use_cache=True`, then evaluates each question branch $q_m$ by reusing the cached key-value tensors and calling `pkv.crop(-len(br_m))` after each branch.
   - Guarantees **exact mathematical isolation** across questions in the same request: adding or reordering sibling questions never changes another question's probability distribution.
3. **All-42-Layer LoRA + Hybrid Proper Scoring Rule Loss**:
   - Attaches LoRA (`r=16, lora_alpha=32, lora_dropout=0.0`) to all 42 Transformer layers across all 7 linear projections (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`, ~25.3M trainable parameters).
   - Trained with a composite proper scoring rule combining KL divergence ($\text{KL}(p_{\text{target}} \,\|\, p_\theta)$), Brier score ($0.25 \cdot \|p_\theta - p_{\text{target}}\|_2^2$), and an ordinal distance penalty ($0.15 \cdot \mathbb{E}[|i - y| / (K-1)]$) for `score` rubrics.

---

## 3. Repository Structure

```text
github-MiniCPM5-2B-Jev/
├── README.md                     # Project overview, architecture & quickstart
├── LICENSE                       # Apache-2.0 License
├── pyproject.toml                # Package configuration & CLI entrypoints
├── requirements.txt              # Python dependencies
├── model.py                      # MiniCPMSystemOne + LetterReadoutHead + wire format converter
├── serve.py                      # FastAPI HTTP server (/v1/systemone & /health)
├── eval_benchmarks.py            # Unified evaluator across all 7 decision benchmark suites
├── train.py                      # All-Layer LoRA trainer & post-hoc calibrator
├── build_stage1_data.py          # 5-Pool System 1 dataset builder (jev_s1_clean_v2)
├── checkpoints/
│   └── stage2/                   # Primary shipped checkpoint (MiniCPM5-2B-Jev)
│       ├── adapter_model.safetensors  # 100.5 MB LoRA weights
│       ├── adapter_config.json        # PEFT configuration
│       ├── head.pt                    # 2.1 MB LetterReadoutHead + calibration parameters
│       ├── checkpoint_meta.json       # Calibration & validation metadata
│       └── benchmark_results.json     # Full evaluation results on all 7 suites
├── benchmarks/
│   ├── stage2_results.json            # MiniCPM5-2B-Jev evaluation metrics
│   ├── stage1_results.json            # Stage 1 (top-20 layer LoRA) baseline metrics
│   ├── training_history.json          # Step 0 -> Step 800 training & calibration log
│   └── dataset_stats.json             # Statistics of the 5-pool jev_s1_clean_v2 dataset
└── data/
    ├── jevbench/                 # Official JevBench public suite (easy, original, hard)
    ├── jabr/                     # jabr classifier-benchmark v1 (78) & v2 (869)
    └── stage2/                   # Held-out validation splits (val_10source, val_hard_v1, dev)
```

---

## 4. Quickstart

### 4.1 Installation

```bash
pip install -r requirements.txt
```

### 4.2 Python Inference API

```python
import torch
from model import MiniCPMSystemOne, to_internal_record

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model, tok = MiniCPMSystemOne.load_checkpoint("checkpoints/stage2", device=device, dtype=torch.bfloat16)

record = {
    "state": "Ticket #402: Customer upgraded from Basic to Enterprise on March 10, 2026 and requests SSO configuration help.",
    "questions": {
        "queue": {
            "type": "choice",
            "instructions": "Route the ticket to the appropriate support queue.",
            "criteria": {
                "billing": "Invoices, refunds, subscription cancellation",
                "enterprise_onboarding": "SSO, SAML, dedicated cluster setup for Enterprise accounts",
                "general_tech": "Bug reports and standard troubleshooting",
            },
        },
        "is_enterprise": {
            "type": "noul",
            "instructions": "Is the customer currently on the Enterprise plan?",
        },
    },
}

internal = to_internal_record(record, add_date_facts=True)
outputs = model.predict_record(internal, use_pride=False)

for q, (_, probs) in zip(internal["questions"], outputs):
    dist = {k: round(float(p), 4) for k, p in zip(q["keys"], probs.tolist())}
    print(f"{q['name']} ({q['type']}): {dist}")
```

### 4.3 HTTP Server (`/v1/systemone`)

```bash
python serve.py --checkpoint checkpoints/stage2 --host 127.0.0.1 --port 8013
```

Test with `curl`:
```bash
curl -s http://127.0.0.1:8013/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "state": "Applicant has 6 years of Python experience, 2 years of Rust, and holds an M.S. in Computer Science.",
    "questions": {
      "meets_python_req": {
        "type": "noul",
        "instructions": "Does the applicant have at least 5 years of Python experience?"
      },
      "seniority_band": {
        "type": "choice",
        "instructions": "Assign the candidate to an interview track.",
        "criteria": {
          "junior": "0-2 years experience",
          "mid": "3-4 years experience",
          "senior": "5+ years experience with advanced degree or systems language background"
        }
      }
    }
  }' | python3 -m json.tool
```

### 4.4 Running the 7-Suite Benchmark Evaluation

All evaluation datasets (`JevBench`, `jabr v1/v2`, `decision-v7` 10-source, `hard-v1` Template 4) are bundled in `data/` (`DecisionBench` downloads automatically from Hugging Face Hub `akhilaaa3/decision-bench`):

```bash
# Run all 7 benchmark suites on checkpoints/stage2
python eval_benchmarks.py --checkpoint checkpoints/stage2

# Run a specific suite (e.g., JevBench or decision-v7)
python eval_benchmarks.py --checkpoint checkpoints/stage2 --only jevbench
```

### 4.5 Reproducing Training (`train.py`)

```bash
# 1. Build the 5-pool jev_s1_clean_v2 dataset (18,604 train / 770 dev records)
python build_stage1_data.py --out_dir data/stage2

# 2. Run All-Layer LoRA training + post-hoc class-balanced calibration
python train.py \
  --data_dir data/stage2 \
  --out_dir checkpoints/stage2_repro \
  --lora_r 16 \
  --lora_targets all \
  --top_k_layers 0 \
  --epochs 2 \
  --max_steps 800 \
  --batch_size 12 \
  --lr 2.8e-5
```

---

## 5. License

This project and its LoRA / readout head weights are licensed under the [Apache-2.0 License](LICENSE).
