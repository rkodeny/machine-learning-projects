# Airbnb Rental Price Prediction

## Overview

Airbnb Rental Price Prediction is a machine learning project designed to estimate property rental prices based on listing characteristics such as location, room type, availability, and accommodation capacity.

The project demonstrates a complete machine learning workflow, including data preprocessing, outlier detection, feature encoding, feature scaling, model training, evaluation, and exploratory data analysis.

The objective is to identify the factors that influence Airbnb pricing and build a predictive model capable of estimating rental prices for new listings.

---

## Dataset

dataSP23.csv -- contains 27,379 AirBnB listings with columns including neighbourhood group, latitude, longitude, room type, price, minimum nights, host listing count, and availability.



### Features

* Neighbourhood Group
* Room Type
* Latitude
* Longitude
* Minimum Nights
* Availability (365 Days)
* Number of Listings by Host
* Other Property Characteristics

### Target Variable

**Price**

The model predicts the rental price of an Airbnb listing based on its features.

---

## Project Structure

```text
Airbnb-Rental-Price-Prediction/
│
├── data/
│   └── data.csv
├── AirBnb.ipynb
├── requirements.txt
└── README.md
```

---

## Project Workflow

### 1. Data Preprocessing

* Dataset loading and inspection
* Removal of irrelevant features
* Missing value analysis
* Duplicate record detection
* Data quality validation

### 2. Outlier Detection

Outliers were identified using the Interquartile Range (IQR) method.

The analysis revealed significant price outliers, which can negatively impact model performance and prediction accuracy.

### 3. Feature Engineering

Categorical variables were transformed into numerical representations using:

* Label Encoding

Encoded features include:

* Neighbourhood Group
* Room Type

### 4. Data Splitting

The dataset was divided into:

* Training Set
* Testing Set

This ensures unbiased evaluation of model performance on unseen data.

### 5. Feature Scaling

Features were standardized using:

* StandardScaler

Scaling improves model stability and training efficiency.

### 6. Model Development

A machine learning model was trained to predict Airbnb rental prices using property and location features.

### 7. Model Evaluation

Model performance was evaluated using:

* R² Score
* Prediction Accuracy Metrics

---

## Exploratory Data Analysis

Several visualizations were created to understand the dataset and identify important trends.

### Price Distribution

* Distribution of rental prices
* Detection of extreme values and skewness

### Feature Relationships

Scatter plots were used to examine relationships between:

* Neighbourhood Group and Price
* Room Type and Price
* Availability and Price

### Summary Statistics

Statistical summaries were generated to understand:

* Central tendency
* Variability
* Data distribution

---

## Key Insights

* Property location significantly impacts rental prices.
* Room type influences pricing patterns across listings.
* The dataset contains several extreme price outliers.
* Availability and host-related features contribute to price variation.
* Proper preprocessing and scaling improve model performance.

---

## Technologies Used

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Data Visualization

* Matplotlib

### Machine Learning

* Scikit-learn

Techniques utilized include:

* Data Cleaning
* Outlier Detection
* Feature Encoding
* Feature Scaling
* Predictive Modeling

---

## Skills Demonstrated

* Exploratory Data Analysis (EDA)
* Data Cleaning and Preprocessing
* Outlier Detection
* Feature Engineering
* Feature Scaling
* Predictive Analytics
* Machine Learning Workflow Development
* Data Visualization

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
AirBnb.ipynb
```

Run all notebook cells sequentially.

---

## Future Improvements

* Replace Logistic Regression with Regression Models
* Implement Linear Regression
* Train Random Forest Regressor
* Compare multiple regression algorithms
* Perform Hyperparameter Optimization
* Deploy the model using Streamlit
* Build an interactive rental price prediction web application

---

## Conclusion

This project demonstrates a complete machine learning workflow for Airbnb rental price prediction. Through data preprocessing, outlier analysis, feature engineering, and predictive modeling, the project provides insights into the factors affecting rental prices and establishes a foundation for developing more advanced property valuation systems.
