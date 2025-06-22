import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image

# ---------------- Load Dataset ----------------
df = pd.read_csv("kidney_disease.csv")

# Convert and clean data
df['pcv'] = pd.to_numeric(df['pcv'], errors='coerce')
df['rc'] = pd.to_numeric(df['rc'], errors='coerce')
df['wc'] = pd.to_numeric(df['wc'], errors='coerce')

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

df['classification'] = df['classification'].str.strip()

# Encode labels
le_target = LabelEncoder()
df['classification'] = le_target.fit_transform(df['classification'])

# Store encoders
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Model training
X = df.drop("classification", axis=1)
y = df["classification"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = XGBClassifier(eval_metric='mlogloss')
model.fit(X_train, y_train)

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="CKD Predictor", layout="centered")
st.title("🔬 Kidney Disease Predictor")

st.markdown("Enter your medical test values below:")

# ---------------- Input Form ----------------
def get_user_input():
    user_data = {}
    numeric_cols = ['age','bp','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc','sg','al','su']
    categorical_cols = {
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

    for col in numeric_cols:
        user_data[col] = st.number_input(col.upper(), min_value=0.0, step=0.1)

    for col, options in categorical_cols.items():
        user_data[col] = st.selectbox(col.upper(), options)

    return user_data

# ---------------- Prediction ----------------
user_input = get_user_input()

if st.button("🔍 Predict"):
    input_df = pd.DataFrame([user_input])

    # Convert numeric
    for col in ['age','bp','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc','sg','al','su']:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)

    # Encode categorical
    for col, le in label_encoders.items():
        try:
            input_df[col] = le.transform([input_df[col][0]])
        except:
            input_df[col] = [0]

    pred = model.predict(input_df)[0]
    pred_label = le_target.inverse_transform([pred])[0]

    # ---------------- Result Display ----------------
    if pred_label.lower() == "ckd":
        st.error("⚠️ Prediction: Kidney Disease Detected!")
        st.image("Kidney_disease_photo.jpg", width=300)
    else:
        st.success("✅ Prediction: No Kidney Disease Detected.")
        st.image("Not_Kidney_disease.jpg", width=300)
