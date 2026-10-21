# Financial Risk: Exploratory Data and Quality Analysis

**Internship project:** This work was completed as part of the Loka Machine Learning Internship. References to "Week X", task numbers, and labs follow the original internship curriculum.

An exploratory study by Atanas Vitanov during the Loka ML internship. The question
is whether the available features and labels support meaningful risk modeling.
This project contains analysis and a presentation; it does not train a risk model.

## Findings

- The dataset contains 15,000 rows, with Low/Medium/High labels split 60%/30%/10%.
- Six columns each have 15% missing values; 9,284 rows have at least one missing value.
- Domain checks identify 11 records with age 18 and 19 years at the current job.
- Correlations and categorical association checks show weak individual feature
  relationships with the target. This raises questions about dataset suitability;
  it does not establish that all possible predictive relationships are absent.

The notebook examines missingness, class imbalance, feature distributions,
correlations, Cramer's V, and statistical versus logical outliers.

## Files and setup

- [Notebook](financial_risk_eda.ipynb): analysis, plots, and saved conclusions.
- [Presentation](Financial_Risk_EDA.pptx.pdf): presentation of the analysis.
- [Data dictionary](data_description.txt): supplied feature descriptions.
- `financial_risk_assessment.csv`: the original dataset used in the notebook.

From this project directory:

```bash
uv sync
uv run jupyter lab financial_risk_eda.ipynb
```

Run the notebook with this directory as its working directory so its relative CSV
path resolves. The included outputs document the original experiment.
