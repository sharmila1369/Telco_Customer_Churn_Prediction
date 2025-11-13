services:
  - type: web
    name: telco-churn-prediction
    runtime: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "gunicorn app:app"
    envVars:
      PYTHON_VERSION: 3.10
