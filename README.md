# 📊 Telecome Customer Churn Analysis Prediction

> A Machine Learning project that predicts whether a customer will churn (leave) or not based on their data.

---

## 📌 Overview

This project focuses on predicting customer churn using machine learning.

The system:
- Processes customer data
- Trains a classification model
- Predicts whether a customer will stay or leave

---

## 🎯 Problem Statement

Customer churn affects business revenue and growth.

Goal:
> Predict whether a customer will churn so that businesses can take preventive actions.

---

## 📂 Dataset

- File: `telco_churn.csv`
- Type: Customer data (categorical + numerical)

### Target Variable
- `Churn` → Yes / No

---

## ⚙️ Project Structure

```
CUSTOMER-CHURN-PREDICTION/
│
├── app/
│   └── app.py
│
├── data/
│   └── telco_churn.csv
│
├── models/
│   ├── churn_model.pkl
│   └── feature_columns.pkl
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── model_evaluation.py
│   ├── feature_importance.py
│   ├── predict.py
│   ├── api.py
│   └── explain_model.py
│
├── requirements.txt
└── README.md
```

---

## 🔄 Workflow

```
Load Data → Preprocessing → Feature Engineering → Model Training → Evaluation → Prediction
```

---

## 🧠 Machine Learning

- Type: Classification
- Model: Trained and saved as `.pkl`
- Output:
  - 0 → No Churn
  - 1 → Churn

---

## 🧪 Model Evaluation

Evaluation is handled in:
```
src/model_evaluation.py
```

Metrics used:
- Accuracy
- Precision
- Recall
- F1 Score

---

## 📊 EDA

Performed in:
```
notebooks/EDA.ipynb
```

Purpose:
- Understand data distribution
- Identify patterns related to churn

---

## 🔧 Core Modules

- `data_preprocessing.py` → Cleaning and preparing data  
- `feature_engineering.py` → Feature transformation  
- `train_model.py` → Model training  
- `predict.py` → Making predictions  
- `api.py` → Serving predictions  
- `feature_importance.py` → Feature analysis  
- `explain_model.py` → Model explanation  

---

## ▶️ How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Train Model

```bash
python src/train_model.py
```

---

### 3. Run Application

```bash
python app/app.py
```

---

## 📈 Output

The system predicts:
- Whether a customer will churn or not

---

## 📄 License

This project is for educational purposes.

---
