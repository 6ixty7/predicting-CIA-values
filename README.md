# CIE L*a*b* Value Prediction Pipeline

This repository trains regression models that predict spectrophotometer CIE
L*a*b* values from RGB, HSV, and image-derived Lab features.

## Run in GitHub Codespaces

Open the repository in Codespaces, then run:

```bash
python -m pip install -r requirements.txt
python model_training_pipeline.py
```

The default dataset is `fabric_color_dataset.csv` beside the script. To use a
different CSV:

```bash
python model_training_pipeline.py path/to/your_dataset.csv
```

## CSV format

The CSV must contain these feature columns:

```text
R_mean,G_mean,B_mean,H,S,V,L_img,a_img,b_img
```

And these spectrophotometer target columns:

```text
L_star,a_star,b_star
```

At least five data rows are required for cross-validation. The included CSV is
a small sample for testing the pipeline; use a larger measured dataset for
meaningful model evaluation.

## Reported metrics

The script evaluates Linear Regression, PLS Regression, Random Forest, and two
XGBoost configurations using RMSE, MAE, R-squared, and CIE76 Delta E.