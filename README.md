# Telco Customer Churn Prediction Using Machine Learning

This project focuses on predicting telecom customer churn using Machine Learning.  
It includes preprocessing, feature engineering, model training, evaluation, and a Streamlit app for real-time churn prediction.

---

## Installation

```bash
pip install -r requirements.txt
To run the Streamlit app:

streamlit run app.py
Skills Used

Python

Numpy

Pandas

Matplotlib

Seaborn

Scikit-learn

Random Forest Classifier

Data Preprocessing

Feature Engineering

Streamlit

Feature Engineering Steps

Data collection

Handling missing values

Encoding categorical variables

Scaling numerical features

Outlier detection

Variable transformation

Selecting best features

Model building

Model evaluation

Saving the model

Deploying using Streamlit
| Model                           | Accuracy  | Precision | Recall   | F1 Score | ROC-AUC  |
| ------------------------------- | --------- | --------- | -------- | -------- | -------- |
| Logistic Regression             | 81.2%     | 0.79      | 0.77     | 0.78     | 0.84     |
| Decision Tree                   | 83.6%     | 0.81      | 0.80     | 0.81     | 0.85     |
| **Random Forest (Final Model)** | **86.5%** | **0.84**  | **0.83** | **0.84** | **0.89** |
Final chosen model: Random Forest

Key Insights

Month-to-month contracts have highest churn

High monthly charges increase churn probability

Electronic check users churn more

Tenure < 12 months → high risk

Long-term contracts reduce churn

Add-on services improve customer retention

Dashboard / App

A Streamlit application is provided to predict churn based on:

Tenure

Contract type

Payment method

Monthly & Total charges

Internet service

Customer demographics

Run locally:

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

Dataset

Telco Customer Churn Dataset (IBM Sample Dataset)

Contains:

Demographics

Account Information

Services Used

Payment Methods

Churn Label

Links

📊 Streamlit App (Local): streamlit run app.py

📝 Notebook: EDA + Model Building

🆘 Support

For any support or questions, contact:
📩 <a href="mailto:sharmilambikavenna@gmail.com">sharmilambikavenna@gmail.com
</a>

🧑‍💻 Author
⭐ V. SHARMILAMBIKA

AI/ML & Data Science Enthusiast

📧 Email: <a href="mailto:sharmila.ai.ds@gmail.com">sharmila.ai.ds@gmail.com
</a>

