import streamlit as st
import shap
import joblib
import pandas as pd
st.set_page_config(
    page_title="PredictEdge AI",
    page_icon="🛢",
    layout="wide"
)
with st.sidebar:

    st.title("🛢️ PredictEdge AI")

    st.write("### Predictive Maintenance")

    st.divider()

    st.markdown("""
### Project

Predict equipment failure using machine learning and industrial sensor data.

---

### Model

🌲 Random Forest Classifier

---

### Developer

**Ali Farooq**

Electrical Engineer

---

Version **1.0**
""")
    st.info("This dashboard is intended for predictive maintenance demonstrations. Predictions are to support inspection and maintenance decisions not replace them.")

st.subheader("📊 Model Performance")
col1, col2, col3, col4 = st.columns(4)

with col1:
     st.metric(
          "Accuracy",
          "98.40%"
     )
with col2:
     st.metric(
         "Precision",
         "85.37%"
     )
with col3:
     st.metric(
          "Recall",
          "57.38%"         
     )
with col4:
     st.metric(
          "F1 Score",
          "68.63%"
     )
st.divider()
st.caption(
     "Performance evaluated on unseen dataset."
     "Metrics are based on Random Forest Classifier."
)

model = joblib.load("models/random_forest_v1.pkl")

# TreeExplainer is SHAPs optimized explainer for tree based models
explainer = shap.TreeExplainer(model)

st.title("🛢️ PredictEdge AI")
st.subheader("Machine Learning Powered Predictive Maintenance Dashboard")
st.markdown(
"""
PredictEdge AI analyzes industrial machine operating parameters and estimates
the probability of machine failure using a Random Forest machine learning model.
The dashboard also explains the AI's decision using SHAP explainability to
support engineering maintenance decisions.
"""
)
st.divider()

left, right = st.columns([3,1])

with left:
     st.subheader("Machine Parameters")

with right:
     st.metric(
          "Model",
          "Random Forest"
     )


feature_importance = model.feature_importances_
feature_names = [
    "Air Temperature",
    "Process Temperature",
    "RPM",
    "Torque",
    "Tool Wear",
    "Type H",
    "Type L",
    "Type M"
]
importance_df = pd.DataFrame({
     "Feature" : feature_names,
     "Importance" :feature_importance
})
importance_df = importance_df.sort_values(
     by="Importance",
     ascending=False
)
st.divider()

left, right = st.columns(2)

with left:
     machine_type = st.selectbox(
      "Machine Type",
     ["L", "M", "H"]
     )

     air_temp = st.number_input(
      "Air Temperature (K)",
       value=300.0
     )

     process_temp = st.number_input(
      "Process Temperature (K)",
     value=310.0
     )

with right:

     rpm = st.number_input(
      "Rotational Speed (rpm)",
      value=1500
     )

     torque = st.number_input(
      "Torque (Nm)",
      value=40.0
     )

     tool_wear = st.number_input(
      "Tool Wear (minutes)",
      value=10
     )

_, center, _ = st.columns([2,2,2])
with center:
     predict = st.button(
          "🔍 Analyze Machine",
          use_container_width=True
     )
if predict:

    type_h = 0
    type_l = 0
    type_m = 0

    if machine_type == "H":
        type_h = 1

    elif machine_type == "L":
        type_l = 1

    else:
        type_m = 1

     # Create the DataFrame
    input_data = pd.DataFrame({
        "Air temperature [K]": [air_temp],
        "Process temperature [K]": [process_temp],
        "Rotational speed [rpm]": [rpm],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear],
        "Type_H": [type_h],
        "Type_L": [type_l],
        "Type_M": [type_m]
    })

    # Make prediction
    # prediction is either 0 or 1
    prediction = model.predict(input_data)

    # SHAP computes how much each feature contributed to the prediction for that one machine
    shap_values = explainer(input_data)
    failure_shap = shap_values.values[0, :, 1]
    feature_names = input_data.columns
    shap_df = pd.DataFrame({
         "Feature": feature_names,
         "SHAP Value": failure_shap
    })
# Absolute Importance
    shap_df["Contribution"] = (
         shap_df["SHAP Value"].abs()
         / shap_df["SHAP Value"].abs().sum()
    ) * 100

# Direction
    shap_df["Direction"] = shap_df["SHAP Value"].apply(
         lambda x: "🔴 Increased Risk" if x > 0
         else "🟢 Reduced Risk"
    )
#Sort
    shap_df = shap_df.sort_values(
          by="Contribution",
         ascending=False
    ) 
    st.subheader("🧠 AI Decision Explanation (SHAP)")
    chart = (
         shap_df
         .set_index("Feature")["Contribution"]
    )
    st.bar_chart(chart)
    
    top3 = shap_df.head(3)
    st.markdown("### Top Contributing Factor")
    for _, row in top3.iterrows():
         
         st.write(
              f"{row['Direction']} **{row['Feature']}** "
              f"({row['Contribution']:.1f}%)"
              
         )
    top_features = ", ".join(
         top3["Feature"].tolist()
    )

    if prediction[0] == 1:

         summary = f"""
    Prediction Outcome:

    Machine Status: High Failure Risk

    Confidence: 92%

    The strongest contributors were:
    {top_features}

    These factors had the greatest influence on the prediction.

    Recommended Maintenance Actions:
    • Inspect bearings.
    • Reduce excessive torque load.
    • Verify shaft alignment.
    • Check lubrication.
    """

    else:

         summary = f"""
    The AI predicts normal machine operation.

    The strongest contributors were:
    {top_features}

    Continue routine inspections.
    """

    st.subheader("🤖 AI Summary")
    st.info(summary)
         

    # Probabilty gives an array with 0,0 entry telling health prob and 0,1 entry telling failure prob
    probability = model.predict_proba(input_data)
    failure_probability = probability[0][1] * 100
    health_probability = probability[0][0] * 100

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Machine Health",
            f"{health_probability:.1f}%"
        )

    with col2:
         st.metric(
        "Failure Probability",
         f"{failure_probability:.1f}%"
         )
    
    if failure_probability < 30:
         risk = "🟢 LOW"

    elif failure_probability < 70:
         risk = "🟡 MEDIUM"

    else:
         risk = "🔴 HIGH"

    with col3:
        st.metric( 
            "Risk Level" ,
            risk
        )

    

    # Show result
    st.divider()

    if prediction[0] == 0:

         st.success("✅ Machine Operating Normally")

    else:

         st.error("⚠️ High Probability of Machine Failure")

    if prediction[0] == 0:

         st.info(
        "Recommendation: Continue normal operation and scheduled inspections."
         )

    else:

         st.warning(
        "Recommendation: Inspect the machine immediately. Check bearings, torque load and tool wear before further operation."
         )
    st.divider()
    st.subheader("📈 Global Feature Importance")

    st.bar_chart(
         importance_df.set_index("Feature")
    )
st.divider()
st.caption(
     "PredictEdge AI v1.0   |  Developed by Ali Farooq   | "
     "Python • Streamlit • Scikit-learn • SHAP"
)