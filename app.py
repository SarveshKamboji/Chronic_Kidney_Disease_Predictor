# Replace the form section in main() with this updated version:

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
