# Loan Approval Prediction using Machine Learning

## Project Overview
This project implements predictive machine learning classification models to determine credit risk profiles and automate loan application approval decisions for financial institutions. Utilizing historical consumer application records, the system learns how an applicant's demographic markers, credit history, and financial standing influence their loan viability. 

The study systematically executes the complete data science lifecycle: data exploration, categorical vector encoding, model fitting using three separate architectures, and rigorous performance evaluation.

---

## Student Details
* **Name:** Ihejirika Chidera Daniel
* **Level:** 300 Level
* **Department:** Software Engineering
* **Matric No:** KDUSEN24029

---

## Dataset Characteristics
The analysis was performed on an independent, non-Titanic tabular collection containing comprehensive credit applicant observations:
* **Dataset Name:** `loan_approval_dataset.csv`
* **Number of Rows:** 45,000 unique records
* **Number of Columns:** 14 features
* **Number of Missing Values:** 0 (Highly complete dataset)

### Features Evaluated
* **Demographics:** Age, Gender, Education
* **Financial Position:** Person Income, Employee Experience, Home Ownership
* **Loan Details:** Loan Amount, Loan Intent, Loan Interest Rate, Loan Percentage
* **Credit Standing:** Credit History, Credit Score, Previous Loan
* **Target Label:** `Loan Status` (0 = Rejected, 1 = Approved)

---

## Preprocessing & Data Preparation
To ensure the mathematical formulas inside the classification models could properly interpret the raw data, the following preprocessing protocols were executed via `pandas` and `scikit-learn`:
1. **Data Cleaning:** Stripped hidden whitespaces from column headers and string observations.
2. **Missing Values:** Handled via programmatic row validation (all 45,000 records preserved due to 0 missing entries).
3. **Categorical Encoding:** Non-numeric text features (`Gender`, `Education`, `Home Ownership`, `Loan Intent`, `Previous Loan`) were mapped into numerical sequences using `LabelEncoder`.
4. **Data Splitting:** Divided the data into a **70% Training Set** to fit model weights and an independent **30% Test Set** to serve as an unbiased benchmark evaluation.

---

## Models Implemented & Evaluation Performance

Three classification algorithms were built, tested, and scored using **Accuracy**, **Precision**, **F1-Score**, and a **Confusion Matrix**:

### 1. Logistic Regression
* **Accuracy:** 0.8790 (87.90%)
* **Precision:** 0.7296
* **F1-Score:** 0.7278
* **Confusion Matrix:**
  ```text
  [[9684  809]
   [ 824 2183]]

