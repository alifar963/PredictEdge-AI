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

Version **3.0**
""")
    st.info("This dashboard is intended for predictive maintenance demonstrations. Predictions are to support inspection and maintenance decisions not replace them.")




manufacturing_model = joblib.load("models/random_forest_v1.pkl")
esp_model = joblib.load("models/esp_model.pkl")
oilwell_model = joblib.load("models/3w_model.pkl")
esp_encoder = joblib.load("models/esp_label_encoder.pkl")

manufacturing_explainer = shap.TreeExplainer(manufacturing_model)
esp_explainer = shap.TreeExplainer(esp_model)
oilwell_explainer = shap.TreeExplainer(oilwell_model)

st.title("🛢️ PredictEdge AI")
if "page" not in st.session_state:
     st.session_state.page = "home"

if st.session_state.page == "home":

     st.subheader("Machine Learning Powered Predictive Maintenance Dashboard")
     st.markdown(
     """
     PredictEdge AI analyzes industrial machine operating parameters and estimates
     the probability of machine failure using a Random Forest machine learning model.
     The dashboard also explains the AI's decision using SHAP explainability to
     support engineering maintenance decisions.
     """
     )
     st.write("Select the Equipment you want to Diagnose.")
     col1, col2, col3 = st.columns(3)
     with col1:
          st.markdown("### ⚙ Manufacturing Machine")
          st.caption("Motor • Bearings • Machine Health")

          if st.button(
            "Open Manufacturing Module",
            use_container_width=True
        ):
            st.session_state.page = "manufacturing"
            st.rerun()
     with col2:
            st.markdown("### 🛢 Electric Submersible Pump")

            st.caption("ESP Vibration Fault Diagnosis")

            if st.button(
              "Open ESP Module",
              use_container_width=True
        ):
              st.session_state.page = "esp"
              st.rerun()
     with col3:
            st.markdown("### ⛽ Oil Well Event Detection")
            st.caption("Petrobras 3W Well Diagnostics")

            if st.button(
              "Open Oil Well Module",
              use_container_width=True
        ):
               st.session_state.page = "oilwell"
               st.rerun()


elif  st.session_state.page == "manufacturing":
     if st.button("⬅ Back"):

           st.session_state.page = "home"

           st.rerun()
  
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

     left, right = st.columns([3,1])

     with left:
          st.subheader("Machine Parameters")

     with right:
          st.metric(
               "Model",
               "Random Forest"
          )


     feature_importance = manufacturing_model.feature_importances_
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
          prediction = manufacturing_model.predict(input_data)

          # SHAP computes how much each feature contributed to the prediction for that one machine
          shap_values = manufacturing_explainer(input_data)
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
          probability = manufacturing_model.predict_proba(input_data)
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

elif st.session_state.page == "esp":
     if st.button ("⬅ Back"):
           st.session_state.page = "home"
           st.rerun()
 

     st.subheader("📊  Model Performance")
     col1, col2, col3, col4 = st.columns(4)
     
     with col1:
          st.metric(
               "Accuracy",
               "94.20%"
          )
     with col2:
          st.metric(
          "Precision",
          "95.27%"
          )
     with col3:
          st.metric(
               "Recall",
               "94.20%"         
          )
     with col4:
          st.metric(
               "F1 Score",
               "94.54%"
          )
     st.divider()
     st.caption(
          "Performance evaluated on unseen dataset."
          "Metrics are based on Random Forest Classifier."
     )
     left, right = st.columns(2)

     with left:

          median_8_13 = st.number_input(
               "Median Vibration (8–13 Hz)",
               value=0.25,
               format="%.4f",
               help="Median vibration amplitude within the 8–13 Hz frequency band."
          )

          rms_98_102 = st.number_input(
               "RMS Vibration (98–102 Hz)",
               value=0.35,
               format="%.4f",
               help="Root Mean Square vibration energy around 100 Hz."
          )

          peak1x = st.number_input(
               "1× Rotational Peak",
               value=1.10,
               format="%.4f",
               help="Amplitude at the shaft rotational frequency."
          )

          feature_a = st.number_input(
               "Feature A",
               value=0.50,
               format="%.4f"
          )

     with right:

          median_98_102 = st.number_input(
               "Median Vibration (98–102 Hz)",
               value=0.18,
               format="%.4f",
               help="Median vibration amplitude around 100 Hz."
          )

          peak2x = st.number_input(
               "2× Rotational Peak",
               value=0.72,
               format="%.4f",
               help="Second harmonic vibration amplitude."
          )

          feature_b = st.number_input(
               "Feature B",
               value=0.42,
               format="%.4f"
          )
     _, center, _ = st.columns([2,2,2])
     with center:
          predict = st.button(
               "🔍 Analyze ESP",
               use_container_width=True
          )
     
     left, right = st.columns([3, 1])
 
     with left:
          st.subheader("🛢 ESP Vibration Parameters")
     with right:
          st.metric("Model", "Random Forest")
 
     feature_importance = esp_model.feature_importances_
     feature_names = [
          "Median (8–13 Hz)",
          "RMS (98–102 Hz)",
          "Median (98–102 Hz)",
          "1× Rotational Peak",
          "2× Rotational Peak",
          "Feature A",
          "Feature B",
     ]
     importance_df = pd.DataFrame({
         "Feature": feature_names,
         "Importance": feature_importance
     })
     importance_df = importance_df.sort_values(
         by="Importance",
         ascending=False
     )
     if predict:
          input_data = pd.DataFrame({
                "median(8,13)": [median_8_13],
                "rms(98,102)": [rms_98_102],
                "median(98,102)": [median_98_102],
                "peak1x": [peak1x],
                "peak2x": [peak2x],
                "a": [feature_a],
                "b": [feature_b]
          })
          st.write(input_data)
          prediction = esp_model.predict(input_data)
          probability = esp_model.predict_proba(input_data)
          prob_df = pd.DataFrame({
               "Condition": esp_encoder.classes_,
               "Probability (%)": probability[0] * 100
          })

          prob_df = prob_df.sort_values(
               by="Probability (%)",
               ascending=False
          )
          st.dataframe(
               prob_df,
               use_container_width=True,
               hide_index=True
          )
          st.subheader("📊 Condition Probability Distribution")

          chart = (
               prob_df
               .set_index("Condition")["Probability (%)"]
          )

          st.bar_chart(chart)


          predicted_class = esp_encoder.inverse_transform(prediction)[0]

          confidence = probability.max() * 100

          # SHAP computes how much each feature contributed to the prediction for that one machine
          shap_values = esp_explainer(input_data)
          failure_shap = shap_values.values[0, :, 1]
          feature_names = input_data.columns
          shap_df = pd.DataFrame({
               "Feature": feature_names,
               "SHAP Value": failure_shap
          })
          display_names = {
               "median(8,13)": "Median (8–13 Hz)",
               "rms(98,102)": "RMS (98–102 Hz)",
               "median(98,102)": "Median (98–102 Hz)",
               "peak1x": "1× Rotational Peak",
               "peak2x": "2× Rotational Peak",
               "a": "Feature A",
               "b": "Feature B"
          }
          shap_df["Feature"] = shap_df["Feature"].replace(display_names)

     
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
     
          # Sort
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
 
          if predicted_class == "Normal":
            summary = f"""
Prediction Outcome:

ESP Status: Healthy Operation

Prediction Confidence: {confidence:.1f}%

The strongest contributors were:

{top_features}

No abnormal vibration pattern detected.

Continue routine vibration monitoring.
"""

          elif predicted_class == "Unbalance":

           summary = f"""
Prediction Outcome:

ESP Status: Unbalance Detected

Prediction Confidence: {confidence:.1f}%

The strongest contributors were:

{top_features}

Recommended Maintenance Actions:

• Inspect rotating assembly.

• Check shaft balance.

• Inspect impeller condition.
"""

          elif predicted_class == "Misalignment":

           summary = f"""
Prediction Outcome:

ESP Status: Misalignment Detected

Prediction Confidence: {confidence:.1f}%

The strongest contributors were:

{top_features}

Recommended Maintenance Actions:

• Verify shaft alignment.

• Inspect coupling condition.
"""

          elif predicted_class == "Rubbing":

           summary = f"""
Prediction Outcome:

ESP Status: Rubbing Detected

Prediction Confidence: {confidence:.1f}%

The strongest contributors were:

{top_features}

Recommended Maintenance Actions:

• Inspect bearings.

• Check rotor clearance.

• Inspect internal wear.
"""

          elif predicted_class == "Faulty sensor":

           summary = f"""
Prediction Outcome:

ESP Status: Faulty Sensor Detected

Prediction Confidence: {confidence:.1f}%

The strongest contributors were:

{top_features}

Recommended Maintenance Actions:

• Verify sensor calibration.

• Inspect wiring.

• Check signal integrity.
"""
 
          st.subheader("🤖 AI Summary")
          st.info(summary)
 
          col1, col2, col3 = st.columns(3)
     
          with col1:
               st.metric(
                    "ESP Condition",
                    predicted_class
               )
          with col2:
               st.metric(
                    "Prediction Confidence",
                    f"{confidence:.1f}%"
               )
     
          if predicted_class == "Normal":
               risk = "🟢 LOW"
          
          else:
               risk = "🔴 HIGH"
     
          with col3:
               st.metric(
                    "Risk Level",
                    risk
               )
 
        # Show result
          st.divider()
     
          if predicted_class == "Normal":
               st.success("✅ ESP Operating Normally")
               st.info( "Recommendation: Continue routine vibration monitoring and scheduled inspections.")
          elif predicted_class == "Unbalance":
               st.error("⚠️ Unbalance Detected")
               st.warning("Recommendation: Inspect rotating assembly, shaft balance and impeller condition.")
          elif predicted_class == "Misalignment":

               st.error("⚠️ Misalignment Detected")

               st.warning(
                    "Recommendation: Inspect shaft alignment and coupling condition."
               )

          elif predicted_class == "Rubbing":

               st.error("⚠️ Rubbing Detected")

               st.warning(
                    "Recommendation: Inspect bearings, rotor clearance and internal wear."
               )

          elif predicted_class == "Faulty sensor":

               st.error("⚠️ Sensor Fault Detected")

               st.warning(
                    "Recommendation: Verify sensor calibration, wiring and signal integrity."
               )
     
         
          st.divider()
          st.subheader("📈 Global Feature Importance")
          st.bar_chart(
               importance_df.set_index("Feature")
          )

elif st.session_state.page == "oilwell":

    if st.button("⬅ Back"):
        st.session_state.page = "home"
        st.rerun()

    st.subheader("⛽ Oil Well Event Detection (Petrobras 3W)")

    st.markdown("""
    Upload a Petrobras 3W well recording CSV file.
    PredictEdge AI will automatically extract statistical features
    and classify the operational event detected in the well.
    """)

    uploaded_file = st.file_uploader(
        "Upload Well CSV File",
        type="csv"
    )

    if uploaded_file is not None:

        raw_df = pd.read_csv(uploaded_file)

        st.success("CSV file loaded successfully!")

        st.write("### Raw Sensor Data Preview")
        st.dataframe(raw_df.head())

        sensor_columns = [
            "P-TPT",
            "T-TPT",
            "P-MON-CKP",
            "T-JUS-CKP"
        ]

        features = {}

        for sensor in sensor_columns:
            features[f"{sensor}_mean"] = raw_df[sensor].mean()
            features[f"{sensor}_std"] = raw_df[sensor].std()
            features[f"{sensor}_min"] = raw_df[sensor].min()
            features[f"{sensor}_max"] = raw_df[sensor].max()
            features[f"{sensor}_median"] = raw_df[sensor].median()

        input_data = pd.DataFrame([features])

        st.write("### Extracted Statistical Features")
        st.dataframe(input_data)
        st.divider()

        prediction = oilwell_model.predict(input_data)
        probabilities = oilwell_model.predict_proba(input_data)

       

        event_names = {
          0: "Normal Operation",
          1: "Abrupt Increase of BSW",
          2: "Spurious Closure of DHSV",
          3: "Severe Slugging",
          4: "Flow Instability",
          5: "Rapid Productivity Loss",
          6: "Quick Restriction in PCK",
          7: "Scaling in PCK",
          8: "Hydrate in Production Line"
         }
        
        predicted_class = int(prediction[0])
        confidence = probabilities[0][predicted_class] * 100
        predicted_event = event_names[predicted_class]
        if predicted_event == "Normal Operation":
          status = "🟢 Healthy"
          priority = "Routine Monitoring"

        elif predicted_event in [
          "Abrupt Increase of BSW",
          "Flow Instability"
        ]:
          status = "🟡 Attention Required"
          priority = "Monitor Closely"

        elif predicted_event in [
          "Rapid Productivity Loss",
          "Quick Restriction in PCK",
          "Scaling in PCK"
        ]:
          status = "🟠 Maintenance Recommended"
          priority = "Schedule Maintenance"

        elif predicted_event in [
          "Severe Slugging",
          "Hydrate in Production Line",
          "Spurious Closure of DHSV"
        ]:
          status = "🔴 Critical"
          priority = "Immediate Intervention"

        else:
          status = "⚪ Unknown"
          priority = "Review Required"


        col1, col2, col3, col4 = st.columns(4)
        st.divider()
                
        with col1:
                    st.metric(
                         "Detected Event",
                         predicted_event
                    )
                
        with col2:
                    st.metric(
                           "Confidence",
                           f"{confidence:.1f}%"
                    )
                
        with col3:
                    st.metric("Operatonal Status", status)
        with col4:
                    st.metric("Priority", priority)
       


        shap_values = oilwell_explainer(input_data)
        event_shap = shap_values.values[0, :, predicted_class]
        shap_df = pd.DataFrame({
          "Feature": input_data.columns,
          "SHAP Value": event_shap
        })
        shap_df["Contribution"] = (
          shap_df["SHAP Value"].abs()
          / shap_df["SHAP Value"].abs().sum()
        ) * 100

        shap_df["Direction"] = shap_df["SHAP Value"].apply(
          lambda x: "🔴 Increased Prediction"
          if x > 0 else "🟢 Reduced Prediction"
        )

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

        st.markdown("### Top Contributing Factors")

        for _, row in top3.iterrows():

         st.write(
          f"{row['Direction']} **{row['Feature']}** "
          f"({row['Contribution']:.1f}%)"
         )

         top_features = ", ".join(
          top3["Feature"].tolist()
         )
        if predicted_event == "Normal Operation":

          summary = f"""
Prediction Outcome:

Well Status: Normal Operation

Prediction Confidence: {confidence:.1f}%

The strongest contributing features were:

{top_features}

The well is operating within expected conditions.

Recommended Actions:

• Continue normal production.
• Maintain routine monitoring.
• Continue scheduled preventive maintenance.
"""

        elif predicted_event == "Abrupt Increase of BSW":

          summary = f"""
Prediction Outcome:

Well Status: Abrupt Increase of BSW

Prediction Confidence: {confidence:.1f}%

The strongest contributing features were:

{top_features}

The model detected an abnormal increase in Basic Sediment & Water (BSW), which may indicate water breakthrough or changes in reservoir behavior.

Recommended Actions:

• Monitor water cut trends.
• Review reservoir production history.
• Inspect separator efficiency.
• Evaluate production optimization options.
"""

        elif predicted_event == "Spurious Closure of DHSV":

          summary = f"""
Prediction Outcome:

Well Status: Spurious Closure of Downhole Safety Valve

Prediction Confidence: {confidence:.1f}%

The strongest contributing features were:

{top_features}

The prediction indicates an unexpected closure of the Downhole Safety Valve.

Recommended Actions:

• Verify hydraulic control pressure.
• Inspect DHSV control system.
• Review shutdown logs.
• Confirm valve functionality before restart.
"""

        elif predicted_event == "Severe Slugging":

          summary = f"""
Prediction Outcome:

Well Status: Severe Slugging

Prediction Confidence: {confidence:.1f}%

The strongest contributing features were:

{top_features}

Large production flow oscillations consistent with severe slugging have been detected.

Recommended Actions:

• Monitor pressure fluctuations.
• Adjust choke settings if appropriate.
• Verify separator stability.
• Continue close production monitoring.
"""

        elif predicted_event == "Flow Instability":

          summary = f"""
Prediction Outcome:

Well Status: Flow Instability

Prediction Confidence: {confidence:.1f}%

The strongest contributing features were:

{top_features}

The production system is exhibiting unstable flow behavior.

Recommended Actions:

• Review pressure trends.
• Inspect production choke.
• Check for changing operating conditions.
• Continue monitoring for escalation.
"""

        elif predicted_event == "Rapid Productivity Loss":

          summary = f"""
Prediction Outcome:

Well Status: Rapid Productivity Loss

Prediction Confidence: {confidence:.1f}%

The strongest contributing features were:

{top_features}

The model indicates a significant reduction in production performance.

Recommended Actions:

• Compare current production with historical trends.
• Investigate reservoir and well performance.
• Inspect surface production equipment.
• Plan engineering evaluation.
"""

        elif predicted_event == "Quick Restriction in PCK":

          summary = f"""
Prediction Outcome:

Well Status: Quick Restriction in Production Choke

Prediction Confidence: {confidence:.1f}%

The strongest contributing features were:

{top_features}

A rapid restriction at the production choke has been detected.

Recommended Actions:

• Inspect the production choke.
• Check for blockage or debris.
• Verify choke valve operation.
• Restore normal flow conditions.
"""

        elif predicted_event == "Scaling in PCK":

          summary = f"""
Prediction Outcome:

Well Status: Scaling in Production Choke

Prediction Confidence: {confidence:.1f}%

The strongest contributing features were:

{top_features}

The prediction suggests scale buildup affecting the production choke.

Recommended Actions:

• Inspect for mineral deposits.
• Review scaling tendency.
• Schedule cleaning if required.
• Consider chemical scale inhibition.
"""

        elif predicted_event == "Hydrate in Production Line":

          summary = f"""
Prediction Outcome:

Well Status: Hydrate Formation

Prediction Confidence: {confidence:.1f}%

The strongest contributing features were:

{top_features}

The model detected conditions consistent with hydrate formation in the production line.

Recommended Actions:

• Verify line temperature and pressure.
• Consider hydrate inhibition procedures.
• Inspect production flow.
• Monitor for blockage development.
"""

        st.subheader("🤖 AI Well Report")
        st.info(summary)
        


st.divider()
st.caption(
     "PredictEdge AI v3.0   |  Developed by Ali Farooq   | "
     "Python • Streamlit • Scikit-learn • SHAP"
)
print(esp_model.classes_)