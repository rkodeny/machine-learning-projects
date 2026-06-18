# Titanic Survival Prediction

## Overview

Titanic Survival Prediction is an end-to-end machine learning project that predicts passenger survival during the Titanic disaster using demographic and travel-related attributes. The project demonstrates the complete data science lifecycle, including exploratory data analysis (EDA), data preprocessing, feature engineering, model development, hyperparameter optimization, and performance evaluation.

The objective is to identify the key factors influencing survival and build predictive models capable of classifying passenger outcomes with high accuracy.

---

## Dataset

**Source:** Kaggle – Titanic: Machine Learning from Disaster

**Dataset Size:** 891 passenger records

### Features

* Passenger Class (Pclass)
* Name
* Sex
* Age
* Number of Siblings/Spouses Aboard (SibSp)
* Number of Parents/Children Aboard (Parch)
* Ticket Number
* Fare
* Cabin
* Embarkation Port (Embarked)

### Target Variable

**Survived**

* 0 → Did Not Survive
* 1 → Survived

---

## Project Structure

```text
Titanic-Survival-Prediction/
│
├── data/
│   └── titanic.csv
├── 01_eda.ipynb
├── 02_data_cleaning.ipynb
├── 03_model_building.ipynb
├── utils.py
├── requirements.txt
└── README.md
```

---

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

* Dataset inspection and profiling
* Missing value analysis
* Distribution analysis of numerical and categorical features
* Correlation analysis
* Survival trends across gender, passenger class, age groups, fare ranges, and family size

### 2. Data Preprocessing & Feature Engineering

* Missing value imputation using statistical methods
* Categorical variable encoding
* Feature scaling
* Creation of engineered features:

  * Title
  * FamilySize
  * IsAlone
  * AgeGroup
  * FareBin
  * HasCabin

### 3. Model Development

Multiple classification algorithms were trained and evaluated:

* Logistic Regression
* Support Vector Machine (SVM)
* Decision Tree
* Random Forest
* Gradient Boosting
* K-Nearest Neighbors (KNN)
* Gaussian Naive Bayes

### 4. Model Optimization

* 5-Fold Cross Validation
* Hyperparameter tuning using GridSearchCV
* Performance comparison using Accuracy, Precision, Recall, and F1-Score

---

## Results

### Model Performance Comparison

| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | 83.80%   | 79.41%    | 78.26% | 78.83%   |
| SVM                 | 82.68%   | 79.69%    | 73.91% | 76.69%   |
| Random Forest       | 81.56%   | 78.12%    | 72.46% | 75.19%   |
| Gradient Boosting   | 81.56%   | 81.03%    | 68.12% | 74.02%   |
| Decision Tree       | 79.89%   | 77.97%    | 66.67% | 71.88%   |
| Naive Bayes         | 77.65%   | 68.83%    | 76.81% | 72.60%   |
| KNN                 | 77.09%   | 75.00%    | 60.87% | 67.20%   |

### Best Performing Model

**Logistic Regression**

* Accuracy: 83.80%
* Precision: 79.41%
* Recall: 78.26%
* F1 Score: 78.83%

### Hyperparameter-Tuned Random Forest

Best Parameters:

```python
max_depth=10
min_samples_leaf=1
min_samples_split=10
n_estimators=200
```

### Cross-Validation Performance

| Model               | Mean CV Accuracy |
| ------------------- | ---------------- |
| Logistic Regression | 81.75%           |
| Gradient Boosting   | 81.33%           |
| SVM                 | 81.04%           |
| Random Forest       | 80.77%           |

---

## Key Insights

* Female passengers exhibited significantly higher survival rates (~74%) compared to male passengers (~19%).
* Passenger class was strongly associated with survival outcomes, with First-Class passengers achieving the highest survival rates.
* Higher ticket fares correlated positively with survival probability.
* Children under the age of 12 demonstrated higher survival rates than adults.
* Small family groups (2–4 members) showed better survival outcomes compared to solo travelers and large families.
* Engineered features such as **Title** and **HasCabin** contributed positively to predictive performance.

---

## Technologies Used

### Programming Language

* Python

### Data Analysis & Visualization

* Pandas
* NumPy
* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn

Techniques utilized include:

* Classification Modeling
* Feature Engineering
* Hyperparameter Optimization
* Cross Validation
* Performance Evaluation

---

## Skills Demonstrated

* Exploratory Data Analysis (EDA)
* Data Cleaning and Preprocessing
* Feature Engineering
* Statistical Analysis
* Classification Algorithms
* Model Evaluation
* Hyperparameter Tuning
* Data Visualization
* Machine Learning Workflow Design

---

## Installation & Usage

### Clone Repository

```bash
git clone <repository-url>
cd Titanic-Survival-Prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch Jupyter Notebook

```bash
jupyter notebook
```

Execute notebooks in the following order:

1. `01_eda.ipynb`
2. `02_data_cleaning.ipynb`
3. `03_model_building.ipynb`

---

## Future Enhancements

* Implement XGBoost and LightGBM models
* Develop an ensemble voting classifier
* Deploy the model using Streamlit
* Containerize the project using Docker
* Build an interactive web interface for real-time predictions

---

## Conclusion

This project demonstrates a complete supervised machine learning workflow, from raw data exploration to model optimization and evaluation. Logistic Regression achieved the strongest overall performance, highlighting the effectiveness of well-engineered features and rigorous preprocessing techniques in predictive modeling.

