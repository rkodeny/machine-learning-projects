# Alzheimer's Disease Detection Using Machine Learning

## Overview

This project develops a machine learning pipeline to predict Alzheimer's disease diagnosis using cognitive assessment scores and brain imaging measurements. The workflow includes data preprocessing, feature engineering, handling class imbalance, feature selection, and training an optimized CatBoost classifier.

The goal is to accurately classify patients into diagnosis categories based on clinical and neuroimaging data.

---

## Dataset

The dataset consists of three Excel sheets:

- **Diagnosis Target** – Patient diagnosis labels
- **Cognitive Scores** – Cognitive assessment measurements
- **Brain Imaging Data** – MRI-derived brain structure measurements

The datasets are merged using:

- `RID` (Patient ID)
- `Test_data` (Assessment Date)

---

## Features Used

### Cognitive Features

- MMSE
- ADAS13
- CDRSB

### Brain Imaging Features

- Hippocampus
- Ventricles
- WholeBrain
- Entorhinal

### Engineered Features

- Hippocampus / WholeBrain
- Ventricles / WholeBrain
- Hippocampus / Entorhinal
- Ventricles / Hippocampus
- Hippocampus / Ventricles
- ADAS13 / MMSE
- CDRSB / MMSE
- CDRSB / ADAS13
- MMSE − ADAS13

---

## Methodology

### Data Preprocessing

- Missing value imputation
- One-hot encoding for categorical variables
- Feature scaling where required

### Class Imbalance Handling

- SMOTE (Synthetic Minority Oversampling Technique)

### Feature Selection

- Mutual Information-based feature selection
- Top 75 features retained

### Model

**CatBoost Classifier**

Key parameters:

- Iterations: 2000
- Depth: 8
- Learning Rate: 0.02
- L2 Regularization: 5
- Loss Function: MultiClass

---

## Evaluation

The model is evaluated using:

- Accuracy
- Balanced Accuracy
- Weighted F1 Score
- ROC-AUC (One-vs-Rest)
- Classification Report
- Confusion Matrix

Cross-validation is performed using:

- Stratified 5-Fold Cross Validation

---

## Project Structure

```text
.
├── AlzheimerDetection.ipynb
├── CSI_7_MAL_2324_CW_resit_data.xlsx
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd alzheimers-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
AlzheimerDetection.ipynb
```

Run all cells sequentially.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Imbalanced-Learn
- CatBoost
- Jupyter Notebook

---

## Future Improvements

- Hyperparameter tuning with Optuna
- SHAP-based model explainability
- Ensemble learning approaches
- Deployment using Streamlit or FastAPI

## Author

Machine Learning Project for Alzheimer's Disease Classification using Cognitive and Neuroimaging Data.