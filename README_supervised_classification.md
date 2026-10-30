# Supervised Classification / Regression Modeling & Tuning

## Overview
This project compares four classification algorithms, tunes hyperparameters with stratified 5-fold cross-validation, evaluates performance, selects a champion model using validation F1-score, and serializes the fitted pipeline.

## Files
- `supervised_classification_modeling_tuning.ipynb` — main Google Colab/Jupyter notebook.
- `classification_modeling_dataset.csv` — synthetic binary classification dataset.
- `README_supervised_classification.md` — project documentation.

## How to run
1. Upload the notebook to Google Colab.
2. Run cells from top to bottom.
3. Upload `classification_modeling_dataset.csv` when prompted.
4. Allow all four GridSearchCV searches to complete.
5. Review the comparison table, ROC curves, classification report, and confusion matrix.
6. Download the generated `champion_model.joblib`.

## Methods
- Preprocessing: median imputation and standard scaling for numerical columns; most-frequent imputation and one-hot encoding for categorical columns.
- Models: Logistic Regression, Random Forest, Gradient Boosting, Extra Trees.
- Tuning: GridSearchCV with StratifiedKFold (5 folds), optimized using F1-score.
- Evaluation: precision, recall, F1-score, ROC-AUC, accuracy, ROC curve, confusion matrix.
- Champion selection: best mean cross-validation F1-score.
- Serialization: joblib.

## Dataset note
The included dataset is synthetic and is for demonstration. Replace it with a permitted real dataset if the assignment explicitly requires one.
