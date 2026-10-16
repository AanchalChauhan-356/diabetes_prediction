import streamlit as st
import pandas as pd
import joblib

# ---------- Page setup ----------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
)

# ---------- Load model (cached so it loads only once) ----------
@st.cache_resource
def load_files():
    model = joblib.load("diabetes_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, feature_columns

model, feature_columns = load_files()

# ---------- Header ----------
st.title("🩺 Diabetes Risk Predictor")
st.markdown("A machine learning **screening aid** that estimates diabetes risk from basic health measurements.")

# ---------- Sidebar: inputs ----------
st.sidebar.header("Patient Details")

pregnancies = st.sidebar.slider("Pregnancies", 0, 20, 1)
glucose = st.sidebar.slider("Glucose (mg/dL)", 40, 250, 120)
blood_pressure = st.sidebar.slider("Blood Pressure (mm Hg)", 20, 130, 70)
skin_thickness = st.sidebar.slider("Skin Thickness (mm)", 5, 100, 20)
insulin = st.sidebar.slider("Insulin (mu U/ml)", 10, 900, 80)
bmi = st.sidebar.slider("BMI", 15.0, 70.0, 25.0, 0.1)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.05, 2.5, 0.5, 0.01)
age = st.sidebar.slider("Age (years)", 18, 100, 30)

predict_clicked = st.sidebar.button("Predict", type="primary", use_container_width=True)

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["Prediction", "About the Model"])

with tab1:
    if predict_clicked:
        input_data = pd.DataFrame(
            [[pregnancies, glucose, blood_pressure, skin_thickness,
              insulin, bmi, dpf, age]],
            columns=feature_columns,
        )

        probability = model.predict_proba(input_data)[0][1]

        # Decide the risk level
        if probability < 0.30:
            level, icon = "Low Risk", "🟢"
            box = st.success
        elif probability < 0.60:
            level, icon = "Moderate Risk", "🟡"
            box = st.warning
        else:
            level, icon = "High Risk", "🔴"
            box = st.error

        # Result cards
        col1, col2 = st.columns(2)
        col1.metric("Risk Level", f"{icon} {level}")
        col2.metric("Diabetes Probability", f"{probability:.0%}")

        st.progress(float(probability))
        box(f"The model estimates a **{probability:.0%}** chance of diabetes: **{level}**.")

        # Show what was entered
        with st.expander("See the values you entered"):
            st.dataframe(input_data, use_container_width=True, hide_index=True)

        st.caption("⚠️ This is a screening aid for learning purposes, not a medical diagnosis. Please consult a doctor.")
    else:
        st.info("👈 Enter the patient details in the sidebar and click **Predict**.")

with tab2:
    st.subheader("How it works")
    st.write(
        "The model is a **Random Forest** (200 decision trees voting together) trained on the "
        "Pima Indians Diabetes dataset of 768 patients. The probability is the share of trees that voted 'diabetes'."
    )

    st.subheader("Which features matter most?")
    importances = pd.Series(model.feature_importances_, index=feature_columns).sort_values()
    st.bar_chart(importances)

    st.subheader("Model performance (test set)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", "74%")
    c2.metric("Recall (diabetes)", "0.69")
    c3.metric("F1 (diabetes)", "0.65")

    st.subheader("Limitations")
    st.write(
        "- Trained on a small dataset of women of Pima heritage aged 21+, so it may not generalise to other groups.\n"
        "- Missing values in the original data were filled with the median.\n"
        "- It is a learning project, not a clinical tool."
    )