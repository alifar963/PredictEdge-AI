# 🛢 PredictEdge AI v2

**AI-Powered Predictive Maintenance Platform for Industrial Equipment**

PredictEdge AI is an interactive machine learning application developed to demonstrate predictive maintenance and fault diagnosis for industrial equipment.

The platform currently supports two independent AI modules:

- ⚙️ Manufacturing Machine Failure Prediction
- 🛢 Electric Submersible Pump (ESP) Fault Diagnosis

Built using **Python, Scikit-learn, SHAP and Streamlit**, PredictEdge AI combines machine learning with explainable AI to provide engineers with transparent maintenance recommendations.

---

# 🚀 Live Demo

🔗 **Live Application**

(https://predictedge-ai-gah5epberv3zyp2j2jjfbw.streamlit.app/)

---

# 📌 Features

## Version 1 – Manufacturing Machine Failure Prediction

Predicts whether an industrial manufacturing machine is likely to fail based on operating conditions.

### Input Parameters

- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear
- Machine Type (H / M / L)

### AI Output

- Machine Health
- Failure Probability
- Risk Level
- SHAP Explainability
- AI Generated Maintenance Summary
- Maintenance Recommendation
- Global Feature Importance

---

## Version 2 – ESP Fault Diagnosis

Diagnoses operating conditions of Electric Submersible Pumps using vibration features.

### Input Parameters

- Median Vibration (8–13 Hz)
- RMS Vibration (98–102 Hz)
- Median Vibration (98–102 Hz)
- 1× Rotational Peak
- 2× Rotational Peak
- Feature A
- Feature B

### AI Output

- ESP Operating Condition
- Prediction Confidence
- Risk Level
- SHAP Explainability
- AI Generated Diagnostic Summary
- Fault-specific Maintenance Recommendation
- Global Feature Importance

Supported Conditions

- ✅ Normal
- ⚠️ Unbalance
- ⚠️ Misalignment
- ⚠️ Rubbing
- ⚠️ Faulty Sensor

---

# 🧠 Explainable AI

PredictEdge AI integrates **SHAP (SHapley Additive Explanations)**.

Instead of providing only a prediction, the system explains:

- Which parameters contributed most
- Whether each parameter increased or reduced risk
- Top contributing features
- AI-generated engineering summary

This improves model transparency and helps engineers understand the reasoning behind each prediction.

---

# 📊 Model Performance

## Manufacturing Model

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

```
PredictEdge AI
│
├── app.py
├── models/
│   ├── random_forest_balanced.pkl
│   ├── esp_model.pkl
│   └── esp_label_encoder.pkl
│
├── train_model.py
├── train_esp_model.py
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

- Additional industrial equipment modules
- Real-time IoT sensor integration
- Condition probability visualization
- PDF maintenance reports
- Cloud deployment with user authentication
- Deep learning models for vibration analysis

---

# 👨‍💻 Author

**Ali Farooq**

Electrical Engineer | Machine Learning Enthusiast | Predictive Maintenance

GitHub:
(Add GitHub Link)

LinkedIn:
(Add LinkedIn Link)

---

## ⭐ If you found this project interesting, consider giving it a star!
