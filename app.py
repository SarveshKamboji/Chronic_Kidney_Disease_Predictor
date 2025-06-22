from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

app = Flask(__name__)
CORS(app)

# Load and preprocess dataset
df = pd.read_csv('kidney_disease.csv')

# Convert string-numeric fields
df['pcv'] = pd.to_numeric(df['pcv'], errors='coerce')
df['rc'] = pd.to_numeric(df['rc'], errors='coerce')
df['wc'] = pd.to_numeric(df['wc'], errors='coerce')

# Fill missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# Strip and encode target
df['classification'] = df['classification'].str.strip()
le_target = LabelEncoder()
df['classification'] = le_target.fit_transform(df['classification'])

print("Target label classes:", le_target.classes_)  # Debug line

# Drop 'id' if present
if 'id' in df.columns:
    df = df.drop('id', axis=1)

# Encode all object columns
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Split and train
X = df.drop('classification', axis=1)
y = df['classification']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = XGBClassifier(eval_metric='mlogloss')
model.fit(X_train, y_train)

# Optional: Evaluate model
y_pred_test = model.predict(X_test)
print(classification_report(y_test, y_pred_test, target_names=le_target.classes_))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_df = pd.DataFrame([data])

    # Convert to numeric
    numeric_columns = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot',
                       'hemo', 'pcv', 'wc', 'rc']
    for col in numeric_columns:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)

    # Encode categorical
    for col, le in label_encoders.items():
        if col in input_df.columns:
            try:
                input_df[col] = le.transform([input_df[col][0]])
            except ValueError:
                input_df[col] = [0]

    print("Model Input:\n", input_df)  # Debug log
    prediction = model.predict(input_df)[0]
    print("Raw Prediction:", prediction)  # Debug log

    predicted_class = le_target.inverse_transform([prediction])[0]
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
