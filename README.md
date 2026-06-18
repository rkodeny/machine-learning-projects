<h1 align="center">Beginner Level Machine Learning Projects</h1>

<p align="center">
  <img src="assets/classification.png" alt="Project Overview" width="150">
  <p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
</p>

## Overview

This repository consists of basic to intermediate level machine learning projects.Each project has its own standalone Jupyter notebook that can be cloned and run immediately.

## Table of Learnings

- **Data Cleaning and Preprocessing** -- preparing real-world messy data
- **Exploratory Data Analysis** -- visualizations and statistical insights
- **Machine Learning** -- classification, regression, and anomaly detection
- **Deep Learning** -- CNNs, transfer learning, and NLP models
- **Computer Vision** -- detection, recognition, and pose estimation

## All Projects

| # | Project | Category | Difficulty |
|---|---------|----------|------------|
| 1 | [Titanic Survival Prediction](Titanic%20Survival%20Prediction) | Classification | Beginner |
| 2 | [Iris Flower Classification](Iris%20Flower%20Classification) | Classification | Beginner |
| 3 | [Customer Churn](Customer%20Churn) | Classification | Beginner |
| 4 | [Heart Failure Prediction](Heart%20Failure%20Prediction) | Classification | Beginner |
| 5 | [Rental Prices of Airbnb](Rental%20Prices%20of%20AirBnb) | Regression | Beginner |
| 6 | [Message Spam Filtering](Message%20Spam%20Filtering) | NLP | Beginner |
| 7 | [Sentiment Analysis](Sentiment%20Analysis) | NLP | Intermediate |

### Prerequisites

- Python 3.9+
- Jupyter Notebook or JupyterLab

### Core Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Additional Libraries (by project type)

| Project Type | Install |
|-------------|---------|
| Deep Learning | `pip install tensorflow keras` |
| Computer Vision | `pip install opencv-python` |
| NLP | `pip install nltk` |
| Object Detection | `pip install ultralytics` |

### Installation
```bash
git clone https://github.com/rkodeny/machine-learning-projects.git  
cd machine-learning-projects  

# Run a project

cd "Iris Flower Classification"   
pip install -r requirements.txt
jupyter notebook
```