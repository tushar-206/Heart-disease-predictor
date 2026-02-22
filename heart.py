import pandas as pd
import ssl
import numpy as np
import streamlit as st
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Heart Risk AI",
    page_icon="❤️",
    layout="centered"
)

@st.cache_resource
def load_model():
    df = pd.read_csv("heart.csv")
    
    selected_features = ['age', 'chol', 'trestbps']
    X = df[selected_features]
    
    target_column = 'target'
    y = df[target_column] 

    combined = pd.concat([X, y], axis=1).dropna()
    X = combined[selected_features]
    y = combined[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model

model = load_model()

with st.sidebar:
    st.header("About this App")
    st.info("This tool uses a Random Forest classifier trained on the UCI Heart Disease dataset to estimate risk factors.")
    st.markdown("---")
    st.write("**Typical Ranges:**")
    st.caption("• Cholesterol: < 200 mg/dL is desirable")
    st.caption("• Blood Pressure: < 120 mm Hg is normal")

st.title("Heart Disease Risk Predictor")
st.markdown("Adjust the sliders below to check the patient's risk profile.")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Age")
    age = st.slider("Patient Age", 1, 100, 45)

with col2:
    st.subheader("Cholesterol")
    cholesterol = st.number_input("mg/dl", 1, 600, 200)

with col3:
    st.subheader("Blood Pressure")
    bp = st.number_input("mm Hg", 50, 250, 120)

st.write("")
st.write("")

c1, c2, c3 = st.columns([1, 2, 1])

with c2:
    check_btn = st.button("Analyze Health Data", type="primary", use_container_width=True)

if check_btn:
    input_data = np.array([[age, cholesterol, bp]])
    prediction = model.predict(input_data)
    
    st.divider()
    
    if prediction[0] == 0:
        st.balloons()
        st.success("Result: Low Risk Profile")
        st.write("The model predicts **no presence** of heart disease based on these metrics.")
    else:
        st.error("Result: Elevated Risk Detected")
        st.warning("The model predicts a **potential presence** of heart disease.")
        st.markdown("""
            **Recommendation:**
            * Consult a cardiologist.
            * Monitor diet and stress levels.
        """)