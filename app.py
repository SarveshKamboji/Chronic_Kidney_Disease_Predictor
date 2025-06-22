import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image

# -------------------- Page Config --------------------
st.set_page_config(page_title="CKD Predictor", layout="centered")

# Custom background CSS
st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #B6FBFF, #83A4D4);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stSlider > label {
            font-weight: 600 !important;
            color: #222;
        }
        .stSelectbox > label {
            font-weight: 600 !important;
            color: #222;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center; color:#1a237e;'>🧬 Chronic Kidney Disease Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Provide test values to predict the likelihood of CKD.</p>", unsafe_allow_html=True)

# -------------------- Load and Train Model --------------------
df = pd.read_csv("kidney_disease.csv")

# Clean numeric columns
df['pcv'] = pd.to_numeric(df['pcv'], errors='coerce')
df['rc'] = pd.to_numeric(df['rc'], errors='coerce')
df['wc'] = pd.to_numeric(df['wc'], errors='coerce')

for col in df.columns:
    df[col] = df[col].fillna(df[col].mode()[0]) if df[col].dtype == 'object' else df[col].fillna(df[col].median())

df['classification'] = df['classification'].str.strip()
le_target = LabelEncoder()
df['classification'] = le_target.fit_transform(df['classification'])

# Encode categorical features
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

# -------------------- Feature Definitions --------------------
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
    'wc': "White Blood Cell Count",
    'rc': "Red Blood Cell Count",
    'htn': "Hypertension",
    'dm': "Diabetes Mellitus",
    'cad': "Coronary Artery Disease",
    'appet': "Appetite",
    'pe': "Pedal Edema",
    'ane': "Anemia"
}

slider_fields = {
    'age': (1, 100), 'bp': (50, 180), 'sg': (1.005, 1.025),
    'al': (0, 5), 'su': (0, 5), 'bgr': (70, 500), 'bu': (1, 200),
    'sc': (0.1, 20), 'sod': (100, 150), 'pot': (2.5, 7.5),
    'hemo': (3, 17), 'pcv': (10, 55), 'wc': (1000, 20000), 'rc': (2.5, 6.5)
}

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

# -------------------- User Inputs --------------------
st.markdown("### 📝 Input Patient Data")
user_input = {}

# Sliders for numeric fields
for field, (min_val, max_val) in slider_fields.items():
    step = 0.001 if isinstance(min_val, float) or isinstance(max_val, float) else 1
    user_input[field] = st.slider(f"{feature_labels[field]}", min_value=min_val, max_value=max_val, step=step, value=(min_val + max_val) / 2)

# Select boxes for categorical fields
for field, options in categorical_fields.items():
    user_input[field] = st.selectbox(f"{feature_labels[field]}", options)

# -------------------- Prediction --------------------
if st.button("🔍 Predict"):
    input_df = pd.DataFrame([user_input])

    # Convert numerics
    for col in slider_fields.keys():
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)

    # Encode categoricals
    for col, le in label_encoders.items():
        try:
            input_df[col] = le.transform([input_df[col][0]])
        except:
            input_df[col] = [0]

    # ✅ Fix missing columns
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[X.columns]

    pred = model.predict(input_df)[0]
    pred_label = le_target.inverse_transform([pred])[0]

    st.markdown("---")
    st.markdown("### 🧾 Prediction Result")

    if pred_label.lower() == "ckd":
        st.error("⚠️ Kidney Disease Detected!")
        st.image("Kidney_disease_photo.jpg", width=350)
    else:
        st.success("✅ No Kidney Disease Detected.")
        st.image("Not_Kidney_disease.jpg", width=350)
