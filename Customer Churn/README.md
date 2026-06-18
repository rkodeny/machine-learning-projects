# Customer Churn Prediction Using Logistic Regression From Scratch

## Overview

Customer churn is one of the most important business metrics for subscription-based companies. This project predicts whether a customer is likely to discontinue a service using a custom implementation of Logistic Regression built entirely from scratch.

Rather than relying on pre-built machine learning models, this project demonstrates the mathematical foundations of Logistic Regression by implementing the core components manually, including the sigmoid function, cost function, gradient descent optimization, and prediction logic.

The trained model is then used to identify potential churn among new customers, enabling proactive customer retention strategies.

---

## Business Problem

Customer acquisition is significantly more expensive than customer retention. Predicting churn allows businesses to:

* Identify at-risk customers
* Improve customer retention strategies
* Reduce revenue loss
* Optimize marketing and support efforts

This project provides a data-driven approach to churn prediction using customer demographics and account-related information.

---

## Dataset

### Historical Customer Dataset

**File:** `customer_churn.csv`

The dataset contains customer information and churn labels.

### Features

* Name
* Age
* Total Purchase Amount
* Account Manager
* Years as Customer
* Number of Sites
* Onboard Date
* Location
* Company

### Target Variable

**Churn**

* 0 → Customer Retained
* 1 → Customer Churned

### Prediction Dataset

**File:** `new_customers_1.csv`

Contains new customer records without churn labels. The trained model predicts churn risk for these customers.

---

## Project Structure

```text
Customer-Churn-Prediction/
│
├── data/
│   ├── customer_churn.csv
│   └── new_customers_1.csv
├── customerChurn.ipynb
├── requirements.txt
└── README.md
```

---

## Project Workflow

### 1. Data Exploration

* Dataset inspection
* Data type validation
* Missing value analysis
* Feature selection

### 2. Data Preprocessing

* Removal of non-predictive columns
* Feature extraction
* Train-Test Split
* Feature Standardization using StandardScaler

### 3. Logistic Regression From Scratch

The model was implemented manually without using Scikit-learn's LogisticRegression class.

Core components developed include:

#### Sigmoid Function

Converts linear outputs into probabilities between 0 and 1.

#### Cost Function

Binary Cross-Entropy Loss used to evaluate model performance.

#### Gradient Descent

Iteratively updates model weights to minimize prediction error.

#### Prediction Function

Converts probabilities into binary churn predictions.

---

## Model Training

The custom logistic regression model was trained using:

* Gradient Descent Optimization
* Standardized Features
* Binary Classification Framework

Model parameters were learned directly from the training data.

---

## Results

### Model Performance

| Metric   | Value |
| -------- | ----- |
| Accuracy | 88.9% |

The model successfully classified customer churn behavior with high predictive accuracy.

### New Customer Predictions

The trained model was used to evaluate previously unseen customer records from:

```text
new_customers_1.csv
```

Predicted churn probabilities and classifications are generated for each customer.

---

## Technologies Used

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning Utilities

* Scikit-learn

  * Train-Test Split
  * StandardScaler

### Custom Machine Learning Implementation

* Sigmoid Function
* Cost Function
* Gradient Descent
* Logistic Regression Classifier

---

## Skills Demonstrated

* Machine Learning Fundamentals
* Logistic Regression Mathematics
* Gradient Descent Optimization
* Binary Classification
* Data Preprocessing
* Feature Scaling
* Predictive Analytics
* Model Evaluation
* Customer Churn Analysis

---

## Installation & Usage

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
customerChurn.ipynb
```

Run all notebook cells sequentially.

---

## Example Use Case

The trained model can be used by marketing agencies and subscription-based businesses to:

* Identify high-risk customers
* Prioritize retention campaigns
* Improve customer lifetime value
* Reduce customer attrition

---

## Future Improvements

* Compare with Scikit-learn Logistic Regression
* Implement Regularization (L1/L2)
* Add Cross-Validation
* Train Ensemble Models (Random Forest, XGBoost)
* Deploy as a Streamlit Web Application
* Integrate Real-Time Customer Scoring

---

## Conclusion

This project demonstrates a complete machine learning workflow for customer churn prediction while emphasizing a deep understanding of Logistic Regression through manual implementation. By building the model from scratch and achieving 88.9% accuracy, the project highlights both practical machine learning skills and a strong grasp of the underlying mathematical concepts.
