# Human Activity Recognition: Research Notes

This notebook studies the UCI Human Activity Recognition Using Smartphones dataset.

The goal is to understand the data, train several machine learning models, compare their results, and explain the final model.

## What is included

- Data loading and basic dataset inspection
- Duplicate column name handling
- Class distribution and baseline accuracy checks
- Feature exploration with plots and correlations
- Scaling checks and dimensionality analysis
- Subject-aware train/test splitting
- PCA, t-SNE, and K-Means clustering
- Binary classification: `DYNAMIC` vs `STATIC`
- Custom Mini-Batch SGD Logistic Regression from scratch
- Comparison with `sklearn.linear_model.SGDClassifier`
- MLflow tracking for the custom model
- 6-class activity classification with several models
- Model evaluation with accuracy, macro F1, reports, and confusion matrices
- Partial Dependence Plots for model interpretation
- Final model card

## Main results

The custom Mini-Batch SGD Logistic Regression model was used for the binary `DYNAMIC` vs `STATIC` task.

- Custom model test accuracy: `99.97%`
- Sklearn SGDClassifier test accuracy: `100.00%`

For the full 6-class activity task, LinearSVC had the strongest held-out test score, while XGBoost was selected as the best overall model for the later evaluation sections because it had the strongest cross-validation result and provides class probabilities.

- LinearSVC test accuracy: `96.20%`
- LinearSVC macro F1-score: `96.29%`
- XGBoost test accuracy: `93.82%`
- XGBoost macro F1-score: `93.75%`

The hardest activities to separate were `SITTING` and `STANDING`.

## Notes from the analysis

- Dynamic and static activities are separated well by motion-variability features.
- The full 6-class task depends more on gravity and orientation features.
- GroupKFold gives a more realistic check for subject-level generalization than StratifiedKFold.
- Partial Dependence Plots help show how important features affect predictions.

## MLflow

The notebook logs manual MLflow runs for the custom Mini-Batch SGD model.

Start the local MLflow server before running the MLflow cells:

```bash
uv run mlflow server --host 127.0.0.1 --port 5000
```

Logged values include:

- `learning_rate`
- `n_epochs`
- `batch_size`
- training loss per epoch
- training accuracy per epoch
- final test accuracy
- final macro F1-score

This makes it easier to compare different batch-size experiments.

## How to run

Install the dependencies from this folder, then open the notebook:

```bash
uv sync
cd notebooks
uv run --project .. jupyter lab human_activity_recognition.ipynb
```

The main notebook is:

```text
human_activity_recognition.ipynb
```

The dataset files should be placed in:

```text
notebooks/data/train.csv
notebooks/data/test.csv
```
