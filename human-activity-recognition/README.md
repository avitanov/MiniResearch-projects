# Human Activity Recognition

**Internship project:** This work was completed as part of the Loka Machine Learning Internship. References to "Week X", task numbers, and labs follow the original internship curriculum.

A classical machine-learning study by Atanas Vitanov during the Loka ML internship,
using the UCI smartphone activity-recognition dataset. It combines a mini-batch SGD
logistic-regression implementation from scratch with six-class model comparisons
and analysis of generalization to unseen subjects.

## Recorded results

| Task/model | Test accuracy | Macro F1 |
|---|---:|---:|
| Six activities: LinearSVC | 96.20% | 96.29% |
| Six activities: XGBoost | 93.82% | 93.75% |
| Dynamic vs. static: custom SGD logistic regression | 99.97% | See notebook |

The binary and six-class tasks have different difficulty and should not be
compared directly. The study includes subject-aware splitting, GroupKFold,
PCA/t-SNE, clustering, MLflow tracking, confusion matrices, and partial dependence.
Sitting and standing remain difficult to distinguish. Results describe this
controlled benchmark and do not establish performance on arbitrary phone placements.

## Run

The dataset CSVs are not included. Obtain the smartphone HAR CSV version with 561
sensor-feature columns, `subject`, and `Activity`, then place the files at:

```text
notebooks/data/train.csv
notebooks/data/test.csv
```

From the project root, install the environment and start MLflow:

```bash
uv sync
uv run mlflow server --host 127.0.0.1 --port 5000
```

In a second terminal, launch the notebook from its directory:

```bash
cd notebooks
uv run --project .. jupyter lab human_activity_recognition.ipynb
```

Keep the kernel working directory at `notebooks/`: the original notebook uses
`data/train.csv` and adds `..` to import the custom implementation from `src/`.

See [research notes](RESEARCH_NOTES.md) for the original detailed overview and
[the notebook](notebooks/human_activity_recognition.ipynb) for saved outputs.
Training may produce local MLflow model artifacts; these are excluded from Git.
