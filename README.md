# 🛢️ PredictEdge AI

### Machine Learning Powered Predictive Maintenance Dashboard

PredictEdge AI is an interactive machine learning web application developed using **Python** and **Streamlit** that predicts the probability of industrial machine failure using operating sensor data.

The application combines predictive analytics with AI explainability (SHAP) to provide engineers with both failure predictions and an explanation of the factors influencing each prediction.

---

## 📌 Project Overview

Industrial equipment failures can result in costly downtime, reduced productivity, and increased maintenance expenses.

PredictEdge AI demonstrates how machine learning can support predictive maintenance by analyzing machine operating parameters and estimating the likelihood of equipment failure before it occurs.

The dashboard enables users to:

- Predict machine failure in real time
- View machine health and failure probability
- Assess operational risk
- Understand AI decisions using SHAP explainability
- Identify the most influential operating parameters
- Support engineering maintenance decisions

---

## 🚀 Features

- Interactive Streamlit dashboard
- Real-time machine failure prediction
- Machine health and failure probability
- Risk level classification (Low / Medium / High)
- SHAP AI Explainability
- Global feature importance visualization
- AI-generated maintenance recommendations
- Clean engineering dashboard interface

---

## 📊 Machine Learning Model

**Algorithm**

- Random Forest Classifier

**Model Performance**

| Metric | Score |
|---------|-------|
| Accuracy | **98.40%** |
| Precision | **85.37%** |
| Recall | **57.38%** |
| F1 Score | **68.63%** |

The deployed model was selected after comparing multiple Random Forest configurations and demonstrated the best overall performance on the test dataset.

---

## 📈 Input Parameters

The model predicts machine failure using the following operating parameters:

- Air Temperature
- Process Temperature
- Rotational Speed (RPM)
- Torque
- Tool Wear
- Machine Type (L / M / H)

---

## 🧠 Explainable AI

PredictEdge AI uses **SHAP (SHapley Additive exPlanations)** to explain every prediction.

Rather than only displaying whether a machine is predicted to fail, the application also identifies:

- Features that increased failure risk
- Features that reduced failure risk
- Relative contribution of each parameter

This improves transparency and helps engineers understand the reasoning behind each prediction.

---

## 🛠️ Technology Stack

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- SHAP
- Joblib

---

## 📂 Dataset

This project uses the **AI4I 2020 Predictive Maintenance Dataset**, which contains simulated industrial machine operating data including:

- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear
- Machine Failure Labels

---

## 📷 Dashboard Preview

*Screenshots will be added here.*

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/PredictEdge-AI.git
```

Move into the project folder

```bash
cd PredictEdge-AI
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 🎯 Future Improvements

- Hyperparameter optimization
- Additional machine learning models (XGBoost / LightGBM)
- Model comparison dashboard
- Predictive maintenance scheduling
- PDF report generation
- Cloud deployment with real-time monitoring

---

## 👨‍💻 Developer

**Ali Farooq**

Electrical Engineer

National University of Sciences and Technology (NUST)

Interested in:

- Artificial Intelligence
- Predictive Maintenance
- Industrial Automation
- Oil & Gas Technologies
- Data Analytics

---

## 📄 License

This project was developed for educational and portfolio purposes.