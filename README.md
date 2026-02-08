Got it 👍 — here is the **raw README.md** ready to copy-paste directly.

```md
# TopoGraph Benchmark (TGB)

NeurIPS Dataset & Benchmark Competition on Graph Learning with Topological Features

This repository provides a secure and reproducible template for running a graph learning competition where humans and LLMs compete on equal footing.

The benchmark is designed to never execute participant code. Participants submit predictions only, which are automatically evaluated and ranked using GitHub Actions.

This makes the competition:
- Safe (no untrusted code execution)
- Fully reproducible
- Suitable for human-vs-LLM evaluation studies

---

## 1. Task Overview

Participants must use the provided **topological moment features**.

The core scientific goal of this benchmark is to study:

> How can topological representations be combined with Graph Neural Networks?

Participants are encouraged to design models that **fuse topology and GNNs** in a principled way.

Possible approaches include (but are not limited to):

- Fusion of moment features with GNN embeddings  
- Attention mechanisms over topological descriptors  
- Multi-branch architectures (Topology branch + GNN branch)  
- Feature gating or conditioning of message passing by moments  
- Meta-learning or adaptive weighting between topology and GNN signals  
- Ensemble or late-fusion strategies  

Pure GNN-only or pure moment-only models are allowed as baselines,  
but **the competition is designed so that better fusion is required to win.**

Only predictions are submitted.

### Public Inputs (data/public/)

moments_all.npy  → Feature matrix (N graphs × d features)  
train_ids.npy    → Training graph indices  
val_ids.npy      → Validation graph indices  
test_ids.npy     → Test graph indices  
labels_train.npy → Training labels  
labels_val.npy   → Validation labels (optional)

### Required Output

Predict labels for all graphs in test_ids.npy.

### Evaluation Metric

Primary metric: Macro-F1 score.

---

## Baseline Results

| Model | Validation Accuracy | Validation Macro-F1 |
|---|---:|---:|
| Moments-SVM | **0.3417** | **0.3281** |
| Moments-MLP | 0.3333 | 0.3289 |
| GNN-only (GCN) | 0.3000 | 0.2677 |
| Hybrid Concat (GCN + Moments) | 0.2583 | 0.2488 |

Key observation: precomputed topological moments already contain strong signal.  
Naive fusion with GNN embeddings does not improve performance.

The challenge is to design better fusion strategies between topology and GNN representations.

---

## 2. Repository Structure

.
├── data/
│   ├── public/
│   │   ├── moments_all.npy
│   │   ├── train_ids.npy
│   │   ├── val_ids.npy
│   │   ├── test_ids.npy
│   │   ├── labels_train.npy
│   │   ├── labels_val.npy
│   │   └── sample_submission.csv
│   └── private/
│       └── labels_all.npy   # used only for scoring
│
├── starter_code/
│   ├── baseline_moments_svm.py
│   ├── baseline_moments_mlp.py
│   ├── baseline_gnn_only.py
│   ├── baseline_hybrid_concat.py
│   ├── utils_io.py
│   └── utils_submit.py
│
├── competition/
│   ├── config.yaml
│   ├── validate_submission.py
│   ├── evaluate.py
│   └── metrics.py
│
├── submissions/
│   ├── README.md
│   └── inbox/
│
├── leaderboard/
│   ├── leaderboard.csv
│   └── leaderboard.md
│
└── .github/workflows/
    ├── score_submission.yml
    └── publish_leaderboard.yml

---

## 3. Submission Format

Participants submit one file:

predictions.csv

Format:

id,y_pred  
0,2  
1,0  
2,5  
...

Rules:
- id must match IDs in data/public/test_ids.npy
- One row per test graph
- y_pred must be an integer class label
- No missing or duplicate IDs

Template available in:
data/public/sample_submission.csv

---

## 4. How to Submit

1. Fork the repository  
2. Create folder:

submissions/inbox/<team_name>/<run_id>/

3. Add:
- predictions.csv
- metadata.json

Example metadata.json:

{
  "team": "example_team",
  "model": "human+llm",
  "llm_name": "gpt-5",
  "notes": "Hybrid topology + GNN fusion model"
}

4. Open a Pull Request to main.

The PR will be automatically scored and the result posted as a comment.

---

## 5. Leaderboard

After merging, submissions are added to:
- leaderboard/leaderboard.csv
- leaderboard/leaderboard.md

Rankings are sorted by descending Macro-F1.

---

## 6. Competition Rules

- No external or private data
- No manual labeling of test data
- Do not modify evaluation scripts
- Unlimited offline training allowed
- Only predictions are submitted

Violations may result in disqualification.

---

## 7. Human vs LLM Research Usage

Recommended protocol:
- Fix a time budget (e.g., 2 hours)
- Fix a submission budget (e.g., 5 runs)
- Record metadata (model, llm_name)
- Compare:
  - Validity rate
  - Best score within K submissions
  - Score vs submission index

---

## 8. Citation

If you use this benchmark in academic work, please cite this repository.

---

## 9. License

MIT License.

---

## Interactive Leaderboard (GitHub Pages)

Enable GitHub Pages:

Settings → Pages → Source = main branch /docs folder

Leaderboard URL:
https://<your-org>.github.io/<repo>/leaderboard.html
```
