import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image

# ---------------- Dataset & Model ----------------
df = pd.read_csv("kidney_disease.csv")
df['pcv'] = pd.to_numeric(df['pcv'], errors='coerce')
df['rc'] = pd.to_numeric(df['rc'], errors='coerce')
df['wc'] = pd.to_numeric(df['wc'], errors='coerce')

for col in df.columns:
    df[col] = df[col].fillna(df[col].mode()[0]) if df[col].dtype == 'object' else df[col].fillna(df[col].median())

df['classification'] = df['classification'].str.strip()
le_target = LabelEncoder()
df['classification'] = le_target.fit_transform(df['classification'])

label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop("classification", axis=1)
y = df["classification"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = XGBClassifier(eval_metric='mlogloss')
model.fit(X_train, y_train)

# ---------------- Streamlit Config ----------------
st.set_page_config(page_title="Kidney Disease Predictor", page_icon="🧬", layout="wide")
st.markdown("<h1 style='text-align:center; color:#5e60ce;'>🔬 Chronic Kidney Disease Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>A smart tool to assess risk of CKD based on clinical parameters.</p>", unsafe_allow_html=True)

# ---------------- Feature Labels ----------------
feature_labels = {
    'age': "Age (years)",
    'bp': "Blood Pressure (mm Hg)",
    'sg': "Specific Gravity",
    'al': "Albumin Level",
    'su': "Sugar Level",
    'rbc': "Red Blood Cells",
    'pc': "Pus Cells",
    'pcc': "Pus Cell Clumps",
    'ba': "Bacteria Presence",
    'bgr': "Blood Glucose Random (mg/dl)",
    'bu': "Blood Urea (mg/dl)",
    'sc': "Serum Creatinine (mg/dl)",
    'sod': "Sodium (mEq/L)",
    'pot': "Potassium (mEq/L)",
    'hemo': "Hemoglobin (gms)",
    'pcv': "Packed Cell Volume",
    'wc': "White Blood Cell Count (cells/cumm)",
    'rc': "Red Blood Cell Count (millions/cumm)",
    'htn': "Hypertension",
    'dm': "Diabetes Mellitus",
    'cad': "Coronary Artery Disease",
    'appet': "Appetite",
    'pe': "Pedal Edema",
    'ane': "Anemia"
}

numeric_fields = ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc']
categorical_fields = {
    'rbc': ['normal', 'abnormal'],
    'pc': ['normal', 'abnormal'],
    'pcc': ['present', 'notpresent'],
    'ba': ['present', 'notpresent'],
    'htn': ['yes', 'no'],
    'dm': ['yes', 'no'],
    'cad': ['yes', 'no'],
    'appet': ['good', 'poor'],
    'pe': ['yes', 'no'],
    'ane': ['yes', 'no']
}

# ---------------- Form Inputs ----------------
st.markdown("### 🧾 Enter Medical Test Values")

col1, col2 = st.columns(2)
user_input = {}

with col1:
    for i, field in enumerate(numeric_fields[:len(numeric_fields)//2]):
        user_input[field] = st.number_input(f"{feature_labels[field]}", step=0.1, min_value=0.0)

with col2:
    for i, field in enumerate(numeric_fields[len(numeric_fields)//2:]):
        user_input[field] = st.number_input(f"{feature_labels[field]}", step=0.1, min_value=0.0)

st.markdown("### 🔠 Select Symptoms and Conditions")

cat_col1, cat_col2 = st.columns(2)

with cat_col1:
    for i, (field, options) in enumerate(list(categorical_fields.items())[:len(categorical_fields)//2]):
        user_input[field] = st.selectbox(f"{feature_labels[field]}", options)

with cat_col2:
    for i, (field, options) in enumerate(list(categorical_fields.items())[len(categorical_fields)//2:]):
        user_input[field] = st.selectbox(f"{feature_labels[field]}", options)

# ---------------- Prediction ----------------
if st.button("🧪 Predict CKD Risk"):
    input_df = pd.DataFrame([user_input])

    # Process numeric
    for col in numeric_fields:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)

    # Encode categorical
    for col, le in label_encoders.items():
        try:
            input_df[col] = le.transform([input_df[col][0]])
        except:
            input_df[col] = [0]

    input_df = input_df[X.columns]  # Ensure order

    prediction = model.predict(input_df)[0]
    result = le_target.inverse_transform([prediction])[0]

    st.markdown("---")
    st.markdown("### 🎯 Prediction Result")

    if result.lower() == "ckd":
        st.error("⚠️ Kidney Disease Detected!")
        st.image("Kidney_disease_photo.jpg", width=350, caption="Chronic Kidney Disease")
    else:
        st.success("✅ No Kidney Disease Detected.")
        st.image("Not_Kidney_disease.jpg", width=350, caption="Healthy Kidneys")

    st.markdown("---")
