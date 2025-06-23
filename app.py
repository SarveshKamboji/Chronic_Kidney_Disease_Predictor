
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('kidney_disease.csv')
        
        # Data cleaning
        df['classification'] = df['classification'].str.strip()
        df = df[df['classification'].isin(['ckd', 'notckd'])]  # Keep only valid classes
        
        # Convert string-numeric fields
        numeric_cols = ['pcv', 'rc', 'wc', 'age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 
                       'sc', 'sod', 'pot', 'hemo']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill missing values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())
        
        # Encode target
        le_target = LabelEncoder()
        df['classification'] = le_target.fit_transform(df['classification'])
        
        # Drop 'id' if present
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
        
        # Encode categorical features
        label_encoders = {}
        for col in df.columns:
            if df[col].dtype == 'object' and col != 'classification':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                label_encoders[col] = le
        
        return df, le_target, label_encoders
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

df, le_target, label_encoders = load_data()

# Train model
@st.cache_resource
def train_model():
    if df is None:
        return None, None
    
    X = df.drop('classification', axis=1)
    y = df['classification']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = XGBClassifier(eval_metric='mlogloss')
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred_test = model.predict(X_test)
    report = classification_report(y_test, y_pred_test, target_names=le_target.classes_, output_dict=True)
    return model, report

model, report = train_model()

# Translations
translations = {
    'en': {
        'title': "Kidney Disease Predictor",
        'predict': "Predict",
        'result': "Result:",
        'prediction_labels': {'ckd': "Kidney Disease Detected", 'notckd': "No Kidney Disease"},
        'fields': {
            'age': "Age", 'bp': "Blood Pressure", 'sg': "Specific Gravity", 'al': "Albumin", 'su': "Sugar",
            'rbc': "Red Blood Cells", 'pc': "Pus Cell", 'pcc': "Pus Cell Clumps", 'ba': "Bacteria",
            'bgr': "Blood Glucose Random", 'bu': "Blood Urea", 'sc': "Serum Creatinine", 'sod': "Sodium",
            'pot': "Potassium", 'hemo': "Hemoglobin", 'pcv': "Packed Cell Volume", 'wc': "WBC Count",
            'rc': "RBC Count", 'htn': "Hypertension", 'dm': "Diabetes Mellitus", 'cad': "Coronary Disease",
            'appet': "Appetite", 'pe': "Pedal Edema", 'ane': "Anemia"
        },
        'options': {
            'rbc': ['normal', 'abnormal'], 'pc': ['normal', 'abnormal'],
            'pcc': ['present', 'notpresent'], 'ba': ['present', 'notpresent'],
            'htn': ['yes', 'no'], 'dm': ['yes', 'no'], 'cad': ['yes', 'no'],
            'appet': ['good', 'poor'], 'pe': ['yes', 'no'], 'ane': ['yes', 'no']
        }
    },
    'hi': {
        'title': "किडनी रोग पूर्वानुमान",
        'predict': "भविष्यवाणी करें",
        'result': "परिणाम:",
        'prediction_labels': {'ckd': "किडनी रोग", 'notckd': "कोई किडनी रोग नहीं"},
        'fields': {
            'age': "आयु", 'bp': "ब्लड प्रेशर", 'sg': "विशिष्ट गुरुत्व", 'al': "एल्ब्युमिन", 'su': "शुगर",
            'rbc': "लाल रक्त कोशिकाएं", 'pc': "पस सेल", 'pcc': "पस सेल क्लंप्स", 'ba': "बैक्टीरिया",
            'bgr': "रक्त ग्लूकोज", 'bu': "रक्त यूरिया", 'sc': "सीरम क्रिएटिनिन", 'sod': "सोडियम",
            'pot': "पोटेशियम", 'hemo': "हीमोग्लोबिन", 'pcv': "पैक्ड सेल वॉल्यूम", 'wc': "श्वेत रक्त कोशिकाएं",
            'rc': "लाल रक्त कोशिकाएं", 'htn': "उच्च रक्तचाप", 'dm': "मधुमेह", 'cad': "कोरोनरी रोग",
            'appet': "भूख", 'pe': "सूजन", 'ane': "अनीमिया"
        },
        'options': {
            'rbc': ['सामान्य', 'असामान्य'], 'pc': ['सामान्य', 'असामान्य'],
            'pcc': ['उपस्थित', 'अनुपस्थित'], 'ba': ['उपस्थित', 'अनुपस्थित'],
            'htn': ['हाँ', 'नहीं'], 'dm': ['हाँ', 'नहीं'], 'cad': ['हाँ', 'नहीं'],
            'appet': ['अच्छी', 'खराब'], 'pe': ['हाँ', 'नहीं'], 'ane': ['हाँ', 'नहीं']
        }
    }
}

# Slider configuration
slider_config = {
    'age': {'min': 1, 'max': 100, 'step': 1, 'value': 45},
    'bp': {'min': 50, 'max': 180, 'step': 1, 'value': 80},
    'sg': {'min': 1.0, 'max': 1.05, 'step': 0.01, 'value': 1.02},
    'al': {'min': 0, 'max': 5, 'step': 1, 'value': 0},
    'su': {'min': 0, 'max': 5, 'step': 1, 'value': 0},
    'bgr': {'min': 70, 'max': 500, 'step': 1, 'value': 120},
    'bu': {'min': 10, 'max': 200, 'step': 1, 'value': 45},
    'sc': {'min': 0.5, 'max': 15.0, 'step': 0.1, 'value': 1.2},
    'sod': {'min': 100, 'max': 200, 'step': 1, 'value': 140},
    'pot': {'min': 2.5, 'max': 10.0, 'step': 0.1, 'value': 4.5},
    'hemo': {'min': 3, 'max': 18, 'step': 0.1, 'value': 12.5},
    'pcv': {'min': 20, 'max': 60, 'step': 1, 'value': 40},
    'wc': {'min': 3000, 'max': 15000, 'step': 100, 'value': 7000},
    'rc': {'min': 2.0, 'max': 8.0, 'step': 0.1, 'value': 4.5}
}

# Sidebar
with st.sidebar:
    st.title("Settings")
    language = st.radio("Select Language", ('English', 'हिंदी'), index=0)
    lang_code = 'en' if language == 'English' else 'hi'
    
    st.markdown("---")
    st.markdown("### Model Performance")
    if report:
        st.metric("Accuracy", f"{report['accuracy']:.1%}")
        st.metric("Precision (CKD)", f"{report['ckd']['precision']:.1%}")
        st.metric("Recall (CKD)", f"{report['ckd']['recall']:.1%}")
    else:
        st.warning("Model not trained properly")
def main():
    st.title(translations[lang_code]['title'])
    
    if model is None or le_target is None:
        st.error("Model failed to load. Please check the data and try again.")
        return
    
    with st.form("prediction_form"):
        # Create tabs for better organization
        tab1, tab2, tab3 = st.tabs(["Basic Information", "Blood Tests", "Other Indicators"])
        
        form_data = {}
        
        with tab1:
            st.header("Basic Information")
            cols = st.columns(2)
            
            with cols[0]:
                form_data['age'] = st.slider(
                    translations[lang_code]['fields']['age'],
                    min_value=slider_config['age']['min'],
                    max_value=slider_config['age']['max'],
                    value=slider_config['age']['value'],
                    step=slider_config['age']['step']
                )
                
                form_data['bp'] = st.slider(
                    translations[lang_code]['fields']['bp'],
                    min_value=slider_config['bp']['min'],
                    max_value=slider_config['bp']['max'],
                    value=slider_config['bp']['value'],
                    step=slider_config['bp']['step']
                )
                
                form_data['sg'] = st.slider(
                    translations[lang_code]['fields']['sg'],
                    min_value=slider_config['sg']['min'],
                    max_value=slider_config['sg']['max'],
                    value=slider_config['sg']['value'],
                    step=slider_config['sg']['step']
                )
                
                form_data['al'] = st.slider(
                    translations[lang_code]['fields']['al'],
                    min_value=slider_config['al']['min'],
                    max_value=slider_config['al']['max'],
                    value=slider_config['al']['value'],
                    step=slider_config['al']['step']
                )
                
                form_data['su'] = st.slider(
                    translations[lang_code]['fields']['su'],
                    min_value=slider_config['su']['min'],
                    max_value=slider_config['su']['max'],
                    value=slider_config['su']['value'],
                    step=slider_config['su']['step']
                )
                
            with cols[1]:
                form_data['rbc'] = st.selectbox(
                    translations[lang_code]['fields']['rbc'],
                    options=translations[lang_code]['options']['rbc']
                )
                
                form_data['pc'] = st.selectbox(
                    translations[lang_code]['fields']['pc'],
                    options=translations[lang_code]['options']['pc']
                )
                
                form_data['pcc'] = st.selectbox(
                    translations[lang_code]['fields']['pcc'],
                    options=translations[lang_code]['options']['pcc']
                )
                
                form_data['ba'] = st.selectbox(
                    translations[lang_code]['fields']['ba'],
                    options=translations[lang_code]['options']['ba']
                )
        
        with tab2:
            st.header("Blood Test Results")
            cols = st.columns(2)
            
            with cols[0]:
                form_data['bgr'] = st.slider(
                    translations[lang_code]['fields']['bgr'],
                    min_value=slider_config['bgr']['min'],
                    max_value=slider_config['bgr']['max'],
                    value=slider_config['bgr']['value'],
                    step=slider_config['bgr']['step']
                )
                
                form_data['bu'] = st.slider(
                    translations[lang_code]['fields']['bu'],
                    min_value=slider_config['bu']['min'],
                    max_value=slider_config['bu']['max'],
                    value=slider_config['bu']['value'],
                    step=slider_config['bu']['step']
                )
                
                form_data['sc'] = st.slider(
                    translations[lang_code]['fields']['sc'],
                    min_value=slider_config['sc']['min'],
                    max_value=slider_config['sc']['max'],
                    value=slider_config['sc']['value'],
                    step=slider_config['sc']['step']
                )
                
                form_data['sod'] = st.slider(
                    translations[lang_code]['fields']['sod'],
                    min_value=slider_config['sod']['min'],
                    max_value=slider_config['sod']['max'],
                    value=slider_config['sod']['value'],
                    step=slider_config['sod']['step']
                )
                
            with cols[1]:
                form_data['pot'] = st.slider(
                    translations[lang_code]['fields']['pot'],
                    min_value=slider_config['pot']['min'],
                    max_value=slider_config['pot']['max'],
                    value=slider_config['pot']['value'],
                    step=slider_config['pot']['step']
                )
                
                form_data['hemo'] = st.slider(
                    translations[lang_code]['fields']['hemo'],
                    min_value=slider_config['hemo']['min'],
                    max_value=slider_config['hemo']['max'],
                    value=slider_config['hemo']['value'],
                    step=slider_config['hemo']['step']
                )
                
                form_data['pcv'] = st.slider(
                    translations[lang_code]['fields']['pcv'],
                    min_value=slider_config['pcv']['min'],
                    max_value=slider_config['pcv']['max'],
                    value=slider_config['pcv']['value'],
                    step=slider_config['pcv']['step']
                )
                
                form_data['wc'] = st.slider(
                    translations[lang_code]['fields']['wc'],
                    min_value=slider_config['wc']['min'],
                    max_value=slider_config['wc']['max'],
                    value=slider_config['wc']['value'],
                    step=slider_config['wc']['step']
                )
        
        with tab3:
            st.header("Other Health Indicators")
            cols = st.columns(2)
            
            with cols[0]:
                form_data['rc'] = st.slider(
                    translations[lang_code]['fields']['rc'],
                    min_value=slider_config['rc']['min'],
                    max_value=slider_config['rc']['max'],
                    value=slider_config['rc']['value'],
                    step=slider_config['rc']['step']
                )
                
                form_data['htn'] = st.selectbox(
                    translations[lang_code]['fields']['htn'],
                    options=translations[lang_code]['options']['htn']
                )
                
                form_data['dm'] = st.selectbox(
                    translations[lang_code]['fields']['dm'],
                    options=translations[lang_code]['options']['dm']
                )
                
                form_data['cad'] = st.selectbox(
                    translations[lang_code]['fields']['cad'],
                    options=translations[lang_code]['options']['cad']
                )
                
            with cols[1]:
                form_data['appet'] = st.selectbox(
                    translations[lang_code]['fields']['appet'],
                    options=translations[lang_code]['options']['appet']
                )
                
                form_data['pe'] = st.selectbox(
                    translations[lang_code]['fields']['pe'],
                    options=translations[lang_code]['options']['pe']
                )
                
                form_data['ane'] = st.selectbox(
                    translations[lang_code]['fields']['ane'],
                    options=translations[lang_code]['options']['ane']
                )
        
        # Submit button at the bottom
        submitted = st.form_submit_button(translations[lang_code]['predict'])
        
        if submitted:
            try:
                # Prepare data for prediction
                input_df = pd.DataFrame([form_data])
                
                # Convert to numeric
                for col in slider_config.keys():
                    if col in input_df.columns:
                        input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)
                
                # Encode categorical features
                for col, le in label_encoders.items():
                    if col in input_df.columns:
                        try:
                            input_df[col] = le.transform(input_df[col])
                        except ValueError:
                            input_df[col] = 0  # Default value if encoding fails
                
                # Make prediction
                prediction = model.predict(input_df)[0]
                predicted_class = le_target.inverse_transform([prediction])[0]
                
                # Display result
                st.markdown("---")
                result_col1, result_col2 = st.columns([1, 3])
                
                with result_col1:
                    try:
                        if predicted_class == 'ckd':
                            st.image("assets/ckd_image.jpg", width=200)
                        else:
                            st.image("assets/healthy_kidney.jpg", width=200)
                    except:
                        st.warning("Could not load result image")
                
                with result_col2:
                    result_text = translations[lang_code]['prediction_labels'].get(predicted_class, predicted_class)
                    color = "red" if predicted_class == 'ckd' else "green"
                    st.markdown(f"""
                    <h2 style='color: {color};'>
                        {translations[lang_code]['result']} {result_text}
                    </h2>
                    """, unsafe_allow_html=True)
                    
                    if predicted_class == 'ckd':
                        st.warning("Consult a nephrologist immediately for further evaluation and treatment.")
                        st.info("""
                        **Recommended Actions:**
                        - Schedule an appointment with a kidney specialist
                        - Monitor blood pressure regularly
                        - Reduce salt and protein intake
                        - Stay hydrated
                        """)
                    else:
                        st.success("No signs of kidney disease detected. Maintain a healthy lifestyle!")
                        st.info("""
                        **Prevention Tips:**
                        - Drink plenty of water
                        - Maintain healthy blood pressure
                        - Control blood sugar if diabetic
                        - Avoid excessive painkiller use
                        """)
            
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
