import pickle
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu



diabetes_model = pickle.load(open('diabetic_model2.sav','rb'))
heart_model = pickle.load(open('heart_model.sav','rb'))
parkison_model = pickle.load(open('parkison_model1.sav','rb'))



with st.sidebar:
    
    selected = option_menu('Multiple Disease Prediction System ',
                           ['Diabeties Prediction','Heart Disease Prediction','Parkinsons Prediction'],
                           
                           
                           default_index= 0)
    
    
#Diabetic prediction App

if (selected == "Diabeties Prediction"):
    st.title("Diabeties Prediction using ML")
    
    colm1,colm2,colm3 = st.columns(3)
    
    with colm1:
    
           Pregnancies = st.text_input('Number of Pregnancies')
    with colm2:
           Glucose = st.text_input('Glucose Level')
    with colm3:
           BloodPressure = st.text_input('Blood Pressure value')
    with colm1:
           SkinThickness = st.text_input('Skin Thickness value')
    with colm2:
           Insulin = st.text_input('Insulin Level')
    with colm3:
           BMI = st.text_input('BMI value')
    with colm1:
           DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with colm2:
           Age = st.text_input('Age of the Person')
    
    
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        if(diagnosis_prediction[0]== 1):
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not Diabetic'
            
        
    st.success(diab_diagnosis)
    
#heart  disease prediction
    
if (selected == "Heart Disease Prediction"):
    st.title("Heart Disease Prediction using ML")
    
      
    colm1,colm2,colm3 = st.columns(3)
    
    with colm1:
        age = st.text_input('Age of person')
        trestbps = st.text_input('Resting Blood Pressure (trestbps)')
        restecg = st.text_input('Resting ECG (restecg)')
        oldpeak = st.text_input('ST depression (oldpeak)')
        thal = st.text_input('Thalassemia (thal)')

    with colm2:
        sex = st.text_input('Gender (0 = Female, 1 = Male)')
        chol = st.text_input('Cholesterol')
        thalach = st.text_input('Max Heart Rate (thalach)')
        slope = st.text_input('Slope of ST segment (slope)')

    with colm3:
        cp = st.text_input('Chest Pain Type (cp)')
        fbs = st.text_input('Fasting Blood Sugar (fbs)')
        exang = st.text_input('Exercise Induced Angina (exang)')
        ca = st.text_input('Major Vessels Colored (ca)')

    # Output placeholder
    heart_diagnosis = ""

    # Prediction button
    if st.button('Heart Disease Test Result'):
        try:
            # Convert all inputs to float
            input_data = [float(age), float(sex), float(cp), float(trestbps), float(chol),
                          float(fbs), float(restecg), float(thalach), float(exang),
                          float(oldpeak), float(slope), float(ca), float(thal)]

            # Reshape for model input
            input_np = np.asarray(input_data).reshape(1, -1)

            # Predict
            prediction = heart_model.predict(input_np)

            if prediction[0] == 1:
                heart_diagnosis = 'The person has heart disease.'
            else:
                heart_diagnosis = 'The person does not have heart disease.'
            st.success(heart_diagnosis)

        except:
            st.error("⚠️ Please fill all fields with valid numeric values.")

#parkison prediction

if (selected == "Parkinsons Prediction"):
    st.title("Parkinson’s Prediction using ML")

    # Define input layout with 5 columns
    colm1, colm2, colm3, colm4, colm5 = st.columns(5)

    # Column 1 inputs
    with colm1:
        mdvp_fo_hz = st.text_input('MDVP:Fo(Hz)')
        mdvp_rap = st.text_input('MDVP:RAP')
        shimmer_apq3 = st.text_input('Shimmer:APQ3')
        hnr = st.text_input('HNR')
        spread1 = st.text_input('Spread1')

    # Column 2 inputs
    with colm2:
        mdvp_fhi_hz = st.text_input('MDVP:Fhi(Hz)')
        mdvp_ppq = st.text_input('MDVP:PPQ')
        shimmer_apq5 = st.text_input('Shimmer:APQ5')
        rpde = st.text_input('RPDE')
        spread2 = st.text_input('Spread2')

    # Column 3 inputs
    with colm3:
        mdvp_flo_hz = st.text_input('MDVP:Flo(Hz)')
        jitter_ddp = st.text_input('Jitter:DDP')
        mdvp_apq = st.text_input('MDVP:APQ')
        dfa = st.text_input('DFA')
        d2 = st.text_input('D2')

    # Column 4 inputs
    with colm4:
        mdvp_jitter_percent = st.text_input('MDVP:Jitter(%)')
        mdvp_shimmer = st.text_input('MDVP:Shimmer')
        shimmer_dda = st.text_input('Shimmer:DDA')
        mdvp_shimmer_db = st.text_input('MDVP:Shimmer(dB)')
        ppe = st.text_input('PPE')

    # Column 5 inputs
    with colm5:
        mdvp_jitter_abs = st.text_input('MDVP:Jitter(Abs)')
        nhr = st.text_input('NHR')

    # Placeholder for diagnosis result
    parkison_diagnosis = ''

    # Button to trigger prediction
    if st.button('Parkinson Test Result'):
        try:
            # Convert inputs to float
            input_data = [
                float(mdvp_fo_hz), float(mdvp_fhi_hz), float(mdvp_flo_hz), float(mdvp_jitter_percent),
                float(mdvp_jitter_abs), float(mdvp_rap), float(mdvp_ppq), float(jitter_ddp),
                float(mdvp_shimmer), float(mdvp_shimmer_db), float(shimmer_apq3), float(shimmer_apq5),
                float(mdvp_apq), float(shimmer_dda), float(nhr), float(hnr), float(rpde),
                float(dfa), float(spread1), float(spread2), float(d2), float(ppe)
            ]

            # Make prediction using your trained model
            parkison_prediction = parkison_model.predict([input_data])

            if parkison_prediction[0] == 1:
                parkison_diagnosis = 'The person has Parkinson’s disease.'
            else:
                parkison_diagnosis = 'The person does not have Parkinson’s disease.'

        except Exception as e:
            st.error("Please enter valid numbers in all fields.")

    # Show result
    st.success(parkison_diagnosis)

    

        
    
    
    
    
    