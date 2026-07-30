# 🛢 PredictEdge AI v3

**AI-Powered Predictive Maintenance & Industrial Fault Diagnosis Platform**

PredictEdge AI is an interactive machine learning application designed to demonstrate predictive maintenance, industrial fault diagnosis, and explainable AI for multiple industrial systems.

Built using **Python, Scikit-learn, SHAP, and Streamlit**, the platform enables engineers to analyze equipment health, understand AI predictions, and receive actionable maintenance recommendations.

---

# 🚀 Live Demo

🔗 **Live Application**

https://predictedge-ai-gah5epberv3zyp2j2jjfbw.streamlit.app/

---

# 📌 Features

PredictEdge AI currently includes **three independent AI modules**.

---

# ⚙ Version 1 — Manufacturing Machine Failure Prediction

Predicts whether an industrial manufacturing machine is likely to fail based on operating conditions.

### Input Parameters

- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear
- Machine Type (H / M / L)

### AI Output

- Machine Health Prediction
- Failure Probability
- Risk Assessment
- SHAP Explainability
- AI Maintenance Summary
- Maintenance Recommendations
- Global Feature Importance

---

# 🛢 Version 2 — Electric Submersible Pump (ESP) Fault Diagnosis

Diagnoses operating conditions of Electric Submersible Pumps using vibration-based features.

### Input Parameters

- Median Vibration (8–13 Hz)
- RMS Vibration (98–102 Hz)
- Median Vibration (98–102 Hz)
- 1× Rotational Peak
- 2× Rotational Peak
- Additional vibration features

### AI Output

- ESP Operating Condition
- Prediction Confidence
- Operational Status
- SHAP Explainability
- AI Diagnostic Summary
- Maintenance Recommendations
- Global Feature Importance

### Supported Conditions

- ✅ Normal
- ⚠️ Unbalance
- ⚠️ Misalignment
- ⚠️ Rubbing
- ⚠️ Faulty Sensor

---

# ⛽ Version 3 — Oil Well Event Detection (Petrobras 3W)

Automatically analyzes raw oil well sensor recordings and classifies operational events using the Petrobras 3W dataset.

Unlike the previous modules, users upload a complete well recording and PredictEdge AI automatically extracts statistical features before performing classification.

### Raw Sensor Inputs

- P-TPT
- T-TPT
- P-MON-CKP
- T-JUS-CKP

### Automatic Feature Engineering

For each sensor, PredictEdge AI extracts:

- Mean
- Standard Deviation
- Minimum
- Maximum
- Median

These engineered features are automatically passed to the machine learning model.

### AI Output

- Detected Well Event
- Prediction Confidence
- Operational Status
- Priority Level
- SHAP Explainability
- Top Contributing Features
- AI Well Report
- Engineering Recommendations

### Supported Events

- ✅ Normal Operation
- Abrupt Increase of BSW
- Spurious Closure of DHSV
- Severe Slugging
- Flow Instability
- Rapid Productivity Loss
- Quick Restriction in PCK
- Scaling in PCK
- Hydrate in Production Line

---

# 🧠 Explainable AI

PredictEdge AI integrates **SHAP (SHapley Additive Explanations)** to explain every prediction.

Instead of providing only a classification, the system identifies:

- Most influential features
- Positive and negative feature contributions
- Top contributing parameters
- Prediction confidence
- AI-generated engineering reports

This improves transparency and helps engineers understand the reasoning behind each prediction.

---

# 📊 Model Performance

## Manufacturing Machine Model

| Metric | Score |
|---------|-------|
| Accuracy | 98.40% |
| Precision | 85.37% |
| Recall | 57.38% |
| F1 Score | 68.63% |

---

## ESP Fault Diagnosis Model

| Metric | Score |
|---------|-------|
| Accuracy | 94.20% |
| Precision | 95.27% |
| Recall | 94.20% |
| F1 Score | 94.54% |

---

## Petrobras 3W Oil Well Model

| Metric | Score |
|---------|-------|
| Accuracy | 96.73% |
| Precision | 96.77% |
| Recall | 96.73% |
| F1 Score | 96.72% |

---

# 🛠 Technology Stack

- Python
- Streamlit
- Scikit-learn
- SHAP
- Pandas
- NumPy
- Matplotlib
- Joblib

---

# 📂 Project Structure

```text
PredictEdge AI
│
├── app.py
├── models/
│   ├── random_forest_balanced.pkl
│   ├── esp_model.pkl
│   ├── esp_label_encoder.pkl
│   └── 3w_model.pkl
│
├── scripts/
│   ├── train_model.py
│   ├── train_esp_model.py
│   ├── feature_engineering_3w.py
│   └── train_3w.py
│
├── requirements.txt
└── README.md
```

---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/PredictEdge-AI.git
```

Navigate into the project

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

# 🎯 Future Improvements

- Wireline Risk Advisor
- Slickline Failure Prediction
- Real-time IoT sensor integration
- PDF engineering reports
- Interactive dashboards
- Cloud database integration
- User authentication
- Deep learning models for industrial diagnostics

---

# 👨‍💻 Author

**Ali Farooq**

Electrical Engineer | Machine Learning Engineer | Predictive Maintenance

GitHub:
https://github.com/alifar963

LinkedIn:
(Add your LinkedIn profile)

---

## ⭐ If you found this project useful, consider giving it a star!
