# 💼 Personal Loan Acceptance Prediction

<div align="center">

## Premium Machine Learning Dashboard for Banking & Loan Intelligence

A complete end-to-end machine learning project that predicts whether a customer is likely to accept a personal loan offer and whether a loan application is likely to be approved or rejected.

</div>

---

## 📌 Project Overview

This project focuses on building a professional machine learning solution for the banking sector.

The main goal is to predict which customers are more likely to accept a personal loan offer. Along with this, a second loan approval dataset is also used to predict whether a customer loan application will be approved or rejected.

The project includes complete data preprocessing, exploratory data analysis, model training, model evaluation, business insights, and a premium Streamlit dashboard for real-time predictions.

---

## 🎯 Project Objective

The objective of this project is to:

- Predict which customers are likely to accept a personal loan offer.
- Analyze customer behavior using banking data.
- Train classification models for loan prediction.
- Extract useful business insights from data.
- Build a premium and professional dashboard where users can enter customer details and get predictions instantly.

---

## 🧠 Problem Statement

Banks usually send personal loan offers to a large number of customers. However, only a small percentage of customers actually accept these offers.

Sending offers to every customer can waste time, money, and marketing resources.

This project solves that problem by using machine learning to identify customers who are more likely to accept personal loan offers.

A second model is also created to predict whether a loan application is likely to be approved or rejected.

---

## 📂 Datasets Used

This project uses two datasets:

### 1. Bank Personal Loan Modelling Dataset

This dataset contains customer banking details such as:

- Age
- Experience
- Income
- Family size
- Credit card average spending
- Education level
- Mortgage
- Securities account
- CD account
- Online banking
- Credit card usage
- Personal loan acceptance status

Target column:

```text
Personal Loan
```

### 2. Loan Approval Prediction Dataset

This dataset contains applicant financial and loan details such as:

- Person age
- Person income
- Employment experience
- Loan amount
- Loan interest rate
- Loan percent income
- Credit history length
- Credit score
- Gender
- Education level
- Home ownership
- Loan intent
- Previous loan default status
- Loan approval status

Target column:

```text
loan_status
```

---

## 🛠️ Tools and Technologies

The following tools and technologies are used in this project:

| Category | Tools |
|---|---|
| Programming Language | Python |
| Data Handling | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn |
| Dashboard | Streamlit |
| Model Saving | Joblib |
| Notebook | Jupyter Notebook |
| Code Editor | VS Code |
| Version Control | Git, GitHub |

---

## 🔁 Project Workflow

The complete workflow of this project is:

1. Created professional project folder structure.
2. Loaded both datasets.
3. Performed dataset understanding.
4. Checked data shape, columns, missing values, and summary statistics.
5. Cleaned and prepared the data.
6. Removed unnecessary columns.
7. Handled incorrect and missing values.
8. Performed exploratory data analysis.
9. Created visualizations for important features.
10. Trained classification models.
11. Evaluated models using accuracy, confusion matrix, and classification report.
12. Saved trained models using Joblib.
13. Built a premium Streamlit dashboard.
14. Extracted business insights.
15. Prepared project for GitHub submission.

---

## 📊 Exploratory Data Analysis

EDA was performed to understand customer behavior and loan-related patterns.

### Personal Loan Dataset EDA

The following features were analyzed:

- Age distribution
- Income distribution
- Education level
- Credit card spending
- CD account
- Online banking
- Personal loan acceptance behavior

### Loan Approval Dataset EDA

The following features were analyzed:

- Person income
- Loan amount
- Credit score
- Loan status
- Previous loan default
- Credit history length
- Loan amount compared with income

---

## 📈 Saved Visualizations

All graphs are saved inside the `reports/figures` folder.

```text
reports/figures/
│
├── age_distribution_personal_loan.png
├── income_distribution_personal_loan.png
├── education_vs_personal_loan.png
├── personal_loan_correlation_heatmap.png
├── income_distribution_loan_status.png
└── loan_amount_vs_income.png
```

---

## 🤖 Machine Learning Models

Two machine learning models were trained in this project.

### Model 1: Personal Loan Acceptance Prediction

| Item | Details |
|---|---|
| Algorithm | Logistic Regression |
| Target Column | Personal Loan |
| Accuracy | 95.5% |
| Purpose | Predict whether a customer will accept a personal loan offer |

### Model 2: Loan Approval Prediction

| Item | Details |
|---|---|
| Algorithm | Decision Tree Classifier |
| Target Column | loan_status |
| Accuracy | 91.05% |
| Purpose | Predict whether a loan application will be approved or rejected |

---

## ✅ Model Evaluation

The models were evaluated using the following metrics:

- Accuracy Score
- Confusion Matrix
- Precision
- Recall
- F1-Score
- Classification Report

### Model Performance Summary

| Model | Algorithm | Accuracy |
|---|---|---|
| Personal Loan Acceptance Prediction | Logistic Regression | 95.5% |
| Loan Approval Prediction | Decision Tree Classifier | 91.05% |

---

## 💡 Business Insights

The following insights were extracted from the data:

- High-income customers are more likely to accept personal loan offers.
- Customers with CD accounts and online banking activity may be more interested in banking products.
- Credit score is an important factor in loan approval.
- Previous loan defaults reduce approval chances.
- Loan amount compared to income is an important risk factor.
- Interest rate and credit history also affect loan approval decisions.
- Banks can target high-income and financially stable customers for better marketing results.
- Applicants with strong credit profiles have better chances of loan approval.

---

## 🖥️ Premium Dashboard

A premium Streamlit dashboard is created for this project.

The dashboard allows users to enter customer or applicant details and get real-time predictions.

### Dashboard Pages

The dashboard contains the following pages:

1. Home
2. Personal Loan Prediction
3. Loan Approval Prediction
4. EDA and Insights
5. Model Performance

### Dashboard Features

- Luxury dark theme UI
- Premium banking style layout
- Real-time personal loan acceptance prediction
- Real-time loan approval prediction
- Interactive Plotly charts
- Business insights section
- Model performance comparison
- Clean and professional design

---

## 📁 Project Structure

```text
personal loan acceptance/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   ├── Bank_Personal_Loan_Modelling.csv
│   │   └── Loan Approval Prediction.csv
│   │
│   └── processed/
│       ├── personal_loan_cleaned.csv
│       └── loan_approval_cleaned.csv
│
├── models/
│   ├── personal_loan_model.pkl
│   ├── personal_loan_columns.pkl
│   ├── loan_approval_model.pkl
│   └── loan_approval_columns.pkl
│
├── notebooks/
│   └── 01_personal_loan_acceptance_prediction.ipynb
│
├── reports/
│   ├── model_results.csv
│   │
│   └── figures/
│       ├── age_distribution_personal_loan.png
│       ├── income_distribution_personal_loan.png
│       ├── education_vs_personal_loan.png
│       ├── personal_loan_correlation_heatmap.png
│       ├── income_distribution_loan_status.png
│       └── loan_amount_vs_income.png
│
├── src/
│   └── train_model.py
│
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run This Project

Follow these steps to run the project on your local system.

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/personal-loan-acceptance-prediction.git
```

### 2. Open Project Folder

```bash
cd personal-loan-acceptance-prediction
```

### 3. Create Virtual Environment

```bash
python -m venv venv
```

### 4. Activate Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

### 5. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 6. Train the Models

```bash
python src/train_model.py
```

### 7. Run the Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

---

## 📦 Requirements

The project requires the following Python libraries:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
plotly
joblib
jupyter
```

---

## 🧾 Important Files

| File | Description |
|---|---|
| `src/train_model.py` | Trains both machine learning models |
| `dashboard/app.py` | Streamlit dashboard application |
| `requirements.txt` | Required Python libraries |
| `README.md` | Project documentation |
| `reports/model_results.csv` | Saved model accuracy results |
| `models/*.pkl` | Saved trained models |
| `notebooks/*.ipynb` | Jupyter Notebook with full analysis |

---

## 🔍 Results

### Personal Loan Acceptance Model

The Logistic Regression model achieved strong performance with an accuracy of 95.5%.

This model helps identify customers who are more likely to accept personal loan offers.

### Loan Approval Model

The Decision Tree Classifier achieved an accuracy of 91.05%.

This model helps predict whether a loan application is likely to be approved or rejected.

---

## 🏦 Business Value

This project can help banks and financial institutions to:

- Improve loan marketing campaigns.
- Target the right customers.
- Reduce unnecessary marketing cost.
- Identify high-potential customers.
- Make data-driven loan approval decisions.
- Improve customer segmentation.
- Reduce financial risk.

---

## 🚀 Future Improvements

Possible future improvements include:

- Use more advanced models such as Random Forest, XGBoost, or Gradient Boosting.
- Add feature importance visualization.
- Add customer segmentation using clustering.
- Improve dashboard UI with more advanced charts.
- Deploy dashboard on Streamlit Cloud.
- Add model explainability using SHAP or LIME.
- Add database support for storing predictions.

---

## 📌 Conclusion

This project successfully demonstrates a complete end-to-end machine learning pipeline for banking loan prediction.

It includes:

- Data loading
- Data cleaning
- Exploratory data analysis
- Data visualization
- Classification model training
- Model evaluation
- Business insight extraction
- Premium dashboard development

The final dashboard allows users to enter customer details and get real-time predictions in a professional and interactive interface.

This project can help banks identify high-potential customers and make better loan-related business decisions.

---

<div align="center">

## ⭐ Project Completed Successfully

Made with Python, Machine Learning, and Streamlit

</div>