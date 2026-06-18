# Sentiment Analysis Using Machine Learning

## Overview

This project performs sentiment analysis on social media text using Natural Language Processing (NLP) and Machine Learning techniques. The objective is to classify text into sentiment categories by transforming raw textual data into numerical features and training multiple machine learning models.

The project demonstrates a complete NLP workflow, including data preprocessing, exploratory data analysis, feature engineering, model training, performance evaluation, and sentiment prediction on new text inputs.

---

## Objectives

* Analyze sentiment patterns across social media platforms and regions.
* Clean and preprocess textual data for machine learning.
* Convert text into numerical representations using TF-IDF.
* Train and compare multiple classification algorithms.
* Identify the best-performing model for sentiment prediction.
* Save trained models for future deployment.

---

## Dataset

sentimentdataset.csv - The dataset contains social media posts along with metadata and sentiment labels.

### Key Features

* Text Content
* Sentiment
* Platform
* Country
* Likes
* Retweets
* Timestamp Information

### Target Variable

**Sentiment**

The model predicts the sentiment category associated with each text entry.

---

## Project Structure

```text
Sentiment-Analysis/
│
├── data/
│   └── sentimentdataset.csv
├── SentimentAnalysis.ipynb
├── best_sentiment_model.pkl
├── tfidf_vectorizer.pkl
├── cleaned_sentiment_dataset.csv
├── requirements.txt
└── README.md
```

---

## Project Workflow

### 1. Data Exploration

* Dataset inspection
* Missing value analysis
* Sentiment distribution analysis
* Platform-wise sentiment analysis
* Country-wise sentiment analysis
* Engagement analysis using Likes and Retweets
* Time-based sentiment trends

### 2. Text Preprocessing

The following preprocessing techniques were applied:

* Lowercase conversion
* URL removal
* Punctuation removal
* Stopword removal
* Whitespace normalization

### 3. Feature Engineering

Text data was transformed using:

**TF-IDF Vectorization**

Configuration:

* Maximum Features: 150,000
* N-Gram Range: (1, 3)
* Minimum Document Frequency: 2

Rare sentiment classes were grouped into an "Other" category to improve model stability.

### 4. Model Training

The following machine learning models were trained and evaluated:

* Logistic Regression
* Multinomial Naive Bayes
* Linear Support Vector Machine (SVM)
* Random Forest Classifier

### 5. Model Evaluation

Models were compared using:

* Accuracy Score
* Classification Report
* Precision
* Recall
* F1-Score

The best-performing model was automatically selected based on test accuracy.

---

## Visualizations

The project includes several visual analyses:

* Sentiment Distribution
* Platform-wise Sentiment Analysis
* Country-wise Sentiment Analysis
* Likes vs Retweets Analysis
* Sentiment Activity by Hour
* Model Accuracy Comparison

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
* WordCloud

### Natural Language Processing

* NLTK
* TF-IDF Vectorizer

### Machine Learning

* Scikit-learn

  * Logistic Regression
  * Multinomial Naive Bayes
  * Linear SVM
  * Random Forest

### Model Persistence

* Joblib

---

## Skills Demonstrated

* Natural Language Processing (NLP)
* Text Preprocessing
* Exploratory Data Analysis
* Feature Engineering
* TF-IDF Vectorization
* Machine Learning Classification
* Model Evaluation
* Data Visualization
* Model Serialization

---

## Running the Project

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
SentimentAnalysis.ipynb
```

Run all cells sequentially.

---

## Example Prediction

```python
predict_sentiment("I absolutely love this product!")
```

Example Output:

```text
Positive
```

---

## Outputs Generated

* Trained Sentiment Model (`best_sentiment_model.pkl`)
* TF-IDF Vectorizer (`tfidf_vectorizer.pkl`)
* Cleaned Dataset (`cleaned_sentiment_dataset.csv`)

---

## Future Improvements

* Deep Learning models (LSTM, GRU)
* Transformer-based models (BERT)
* Streamlit deployment
* Real-time social media sentiment monitoring
* Sentiment dashboard with interactive visualizations

---

## Conclusion

This project demonstrates a complete machine learning pipeline for sentiment classification. By combining NLP preprocessing techniques with TF-IDF feature extraction and multiple classification algorithms, the project provides an effective framework for analyzing sentiment in social media text and predicting sentiment on unseen data.
