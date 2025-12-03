# 📞 Telco Customer Churn Prediction Using Machine Learning  
An end-to-end AI-powered churn prediction system that identifies customers likely to leave a telecom service provider.  
This project includes data preprocessing, feature engineering, model building, performance comparison, and a user-friendly prediction app.

---

## 📌 Project Overview
Telecom companies lose revenue when customers stop using their services (churn).  
This project predicts churn using the **Telco Customer Churn dataset**, applying multiple ML models and generating actionable insights.

The workflow includes:
- Exploratory Data Analysis (EDA)
- Feature Engineering & Transformations
- Handling missing values (KNN, Random Sampling, Iterative, Quantile Transform)
- Model training & evaluation
- Deployment with Streamlit

---

## 📂 Dataset
- **Rows:** 7043  
- **Columns:** 21  
- Contains customer demographics, contract details, payment info, services used, and churn label.

Main preprocessing:
- Dropped `customerID`
- Categorical encoding (Label & One-Hot)
- Scaling numeric features (StandardScaler)
- Imputation of missing values
- Variable transformations (Log, Power, RankGauss)

---

## 🧠 Machine Learning Models
The following models were trained and compared:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Logistic Regression | 81.2% | 0.79 | 0.77 | 0.78 | 0.84 |
| Decision Tree | 83.6% | 0.81 | 0.80 | 0.81 | 0.85 |
| **Random Forest (Final Model)** | **86.5%** | **0.84** | **0.83** | **0.84** | **0.89** |

### ✅ Final Production Model: **Random Forest**
Random Forest was selected as the final model because:

- It achieved the **highest stable accuracy (86.5%)**
- It generalizes well and avoids overfitting  
- It provides strong feature importance insights  
- It performs better than Logistic Regression and Decision Tree  
- It is robust for real-world deployment

## 🔍 Key Business Insights
- Month-to-month contract → highest churn  
- High monthly charges → high churn risk  
- Electronic check → most unstable payment group  
- Tenure < 12 months → ~47% churn  
- Long-term contracts → lower churn  
- Auto-pay customers → more loyal  
- Add-on services (Security, TechSupport) reduce churn  

## 📊 Dashboard / Deployment
A Streamlit web app is included to predict churn instantly based on:
- Tenure  
- Contract type  
- Payment method  
- Monthly & Total charges  
- Internet service details  
- Demographic info  

Run locally:
```bash
streamlit run app.py
Project Structure
Telco_Customer_Churn_Prediction/
│
├── app/
│   └── app.py
├── models/
│   ├── churn_model.pkl
│   └── scaler.pkl
├── notebooks/
│   └── EDA_and_Model_Building.ipynb
├── data/
│   └── telco_customer_churn.csv
├── requirements.txt
└── README.md
How to Install & Run
1. Install dependencies
pip install -r requirements.txt

2. Run Streamlit App
streamlit run app.py

🛠 Technologies Used

Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-Learn

XGBoost

SHAP

Streamlit

⭐ Future Enhancements

Deploy on AWS / Azure / Render

Include Deep Learning (ANN) model

Build a SHAP explainability dashboard

Add automated retraining pipeline

---

## 👩‍💻 Author
 V. Sharmilambika 
AI/ML & Data Science Enthusiast  
Email: sharmilambika@gmail.com  


