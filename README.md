Perfect 👍 — here is the **FULL README.md** ready to copy-paste directly into your repo.

Just open `README.md`, delete everything, paste this, save, commit.

---

```md
# TopoGraph Benchmark (TGB)

NeurIPS Dataset & Benchmark Competition on Graph Learning with Topological Features

This repository provides a **secure, reproducible template** for running a
Graph Neural Network (GNN) competition that supports **humans and LLMs**
competing on equal footing.

The design intentionally **does not execute participant code**. Instead,
participants submit **predictions only**, which are automatically evaluated
and ranked on a public leaderboard using GitHub Actions.

This makes the competition:

- Safe (no untrusted code execution)
- Fully reproducible
- Suitable for human-vs-LLM evaluation studies

---

## 1. Task Overview

**Task:** Graph classification using **precomputed topological moment features**.

Participants receive precomputed topological moment features extracted from graphs.
The goal is to build models that predict the **graph class label** for unseen test graphs.

This benchmark evaluates how well models can combine **Graph Neural Networks**
and **Topological Representations**.

### Public Inputs

Located in `data/public/`:

- `moments_all.npy` → Feature matrix (N graphs × d features)
- `train_ids.npy` → Indices of training graphs
- `val_ids.npy` → Indices of validation graphs
- `test_ids.npy` → Indices of test graphs
- `labels_train.npy` → Labels for training graphs
- `labels_val.npy` *(optional)* → Labels for validation graphs

### Output (Submission)

Predict the class label for each graph in `test_ids.npy`.

### Evaluation Metric

**Primary metric:** Macro-F1 score

Participants may train **any model offline**:
- GNNs
- Classical ML
- Deep learning
- Hybrid topology + GNN models
- Ensembles

Only predictions are submitted.

---

## Baseline Results

All baselines are deterministic and provided in `starter_code/`.

| Model | Validation Accuracy | Validation Macro-F1 |
|---|---:|---:|
| Moments-SVM | **0.3417** | **0.3281** |
| Moments-MLP | 0.3333 | 0.3289 |
| GNN-only (GCN) | 0.3000 | 0.2677 |
| Hybrid Concat (GCN + Moments) | 0.2583 | 0.2488 |

### Key Observation

Precomputed **topological moments already contain strong signal**.  
Naive fusion with GNN embeddings **does not improve performance**.

👉 The challenge is to design **better fusion strategies** between topology and GNN representations.

---

## 2. Repository Structure

```

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
│       └── labels_all.npy   # NEVER COMMITTED (used only for scoring)
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

```

---

## 3. Submission Format

Participants submit **one CSV file**:

```

predictions.csv

```

### Format

```

id,y_pred
0,2
1,0
2,5
...

```

### Rules

- `id` must match the IDs in `data/public/test_ids.npy`
- One row per test graph
- `y_pred` must be an **integer class label**
- No missing or duplicate IDs

A template is provided in:

```

data/public/sample_submission.csv

```

---

## 4. How to Submit

1. Fork this repository
2. Create a new folder:

```

submissions/inbox/<team_name>/<run_id>/

````

3. Add:

- `predictions.csv`
- `metadata.json`

### Example metadata.json

```json
{
  "team": "example_team",
  "model": "human+llm",
  "llm_name": "gpt-5",
  "notes": "Hybrid topology + GNN fusion model"
}
````

4. Open a Pull Request to `main`

The PR will be **automatically scored** and the result posted as a comment.

---

## 5. Leaderboard

After a PR is merged, the submission is added to:

* `leaderboard/leaderboard.csv`
* `leaderboard/leaderboard.md`

Rankings are sorted by **descending Macro-F1**.

---

## 6. Rules

* No external or private data
* No manual labeling of test data
* No modification of evaluation scripts
* Unlimited offline training allowed
* Only predictions are submitted

Violations may result in disqualification.

---

## 7. Human vs LLM Research Usage

To use this competition for research:

* Fix a time budget (e.g., 2 hours)
* Fix a submission budget (e.g., 5 runs)
* Record metadata (`model`, `llm_name`)
* Compare:

  * Validity rate
  * Best score within K submissions
  * Score vs submission index

---

## 8. Citation

If you use this benchmark in academic work, please cite the repository.

---

## 9. License

MIT License.

---

## Interactive Leaderboard (GitHub Pages)

Enable GitHub Pages:

Settings → Pages → Source = `main` branch `/docs` folder

Leaderboard will appear at:

```
https://<your-org>.github.io/<repo>/leaderboard.html
```

```

---

Next step: we generate **sample_submission.csv** automatically from `test_ids.npy`.
```
