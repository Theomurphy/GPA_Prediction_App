import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Importing linear regression trained model
model = joblib.load("C:/Users/Trois/Downloads/Video/Data Science/Project/Dataset/Submitted/Corrected/GiutHub_corrected/Model/lr_model.pkl")
feature_order = joblib.load("C:/Users/Trois/Downloads/Video/Data Science/Project/Dataset/Submitted/Corrected/GiutHub_corrected/Model/data_features.pkl")

st.title("🎓 Smart GPA Predictor", width="stretch")
st.markdown("""
Use this intelligent system to predict a student's GPA based on academic and personal information.
""")


# Sidebar customization
with st.sidebar:
    st.markdown("### GPA overview")
    st.caption("**Grade Point Average (GPA), is a numerical representation of a student's academic performance**")
    st.image("C:/Users/Trois/Downloads/Video/Data Science/Project/Dataset/Submitted/Corrected/GPA.PNG")
    st.markdown("---")
    st.caption("**Model Used:** Linear Regression")
    st.caption("📅 Developed: July 2025")
    st.markdown("---")

# One-Hot Mappings
one_hot_race = {'Race_Black':0, 'Race_Hispanic':0, 'Race_Other':0, 'Race_Two-or-more':0, 'Race_White':0}
one_hot_parenteducation = {'ParentalEducation_Bachelors+':0,'ParentalEducation_HS':0, 'ParentalEducation_SomeCollege':0}
one_hot_local = {'Locale_Rural':0 ,'Locale_Suburban':0, 'Locale_Town':0}


with st.form("student_form"):
    st.subheader("📝 Enter Student Details")

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Student name (Optional)", value='')
        Age = st.number_input("Age", 14, 18, value=16)
        Grade = st.selectbox("Grade Level", [9, 10, 11, 12])
        SES_Quartile = st.selectbox("Socioeconomic status ", [1, 2, 3, 4])
        StudyHours = st.slider("Study Hours per Day", 0, 3, value=2)
        TestScore_Math = st.slider("Math Score", 0, 100, value=70)
        TestScore_Reading = st.slider("Reading Score", 0, 100, value=75)
        TestScore_Science = st.slider("Science Score", 0, 100, value=72)

    with col2:
        AttendanceRate = st.slider("Attendance Rate (%)", 0, 100, value=90)
        AttendanceRate = (AttendanceRate/100)
        FreeTime = st.slider("Free Time (h/day)", 0, 5, value=2)
        GoOut = st.slider("Going Out Frequency (Time/week)", 0, 5, value=2)
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        SchoolType = st.radio("School Type", ["Public", "Private"], horizontal=True)
        parental_edu = st.selectbox("Parental Education", ['<HS', 'HS', 'College', 'Bachelors+'])
        race = st.selectbox("Ethnicity", ['Asian', 'Black', 'Hispanic', 'Other', 'Two-or-more', 'White'])
        locale = st.selectbox("Locale", ['Suburban', 'City', 'Rural', 'Town'])

    with st.expander("Additional Factors That could influence the GPA Level"):
        InternetAccess = st.radio("Internet Access", ["Yes", "No"], horizontal=True)
        PartTimeJob = st.radio("Part-Time Job", ["Yes", "No"], horizontal=True)
        Extracurricular = st.radio("Extracurricular Activities", ["Yes", "No"], horizontal=True)
        ParentSupport = st.radio("Parental Support", ["Yes", "No"], horizontal=True)
        Romantic = st.radio("Romantic Relationship", ["Yes", "No"], horizontal=True)

    submit = st.form_submit_button("🎯 Predict GPA")


if submit:
    
    # Manual label Encoding 
    def bin_encode(x):
        return 1 if x == "Yes" or x == "Public" or x == "Male" else 0

    Gender = bin_encode(gender)
    SchoolType = bin_encode(SchoolType)
    InternetAccess = bin_encode(InternetAccess)
    PartTimeJob = bin_encode(PartTimeJob)
    Extracurricular = bin_encode(Extracurricular)
    ParentSupport = bin_encode(ParentSupport)
    Romantic = bin_encode(Romantic)

    # One-Hot Encoding 
    one_hot_race = {key:  1 if race in key else 0 for key in one_hot_race}
    
    one_hot_parenteducation = {key: 1 if (parental_edu in key) else 0 for key in one_hot_parenteducation}
    
    one_hot_local = {key: 1 if (locale in key) else 0 for key in one_hot_local}
            

    # Reconstructing the data
    input_data = pd.DataFrame({'Age': [Age], 'Grade': [Grade],'Gender': [Gender],'SES_Quartile':[SES_Quartile], 'SchoolType': [SchoolType],'TestScore_Math':[TestScore_Math], 
                            'TestScore_Reading':[TestScore_Reading], 'TestScore_Science':[TestScore_Science],'AttendanceRate': [AttendanceRate], 'StudyHours':[StudyHours], 
                            'InternetAccess': [InternetAccess],'Extracurricular': [Extracurricular], 'PartTimeJob': [PartTimeJob],
                            'ParentSupport': [ParentSupport],'Romantic':[Romantic],'FreeTime': [FreeTime],'GoOut':[GoOut], 'Race_Black':[one_hot_race['Race_Black']], 
                            'Race_Hispanic': [one_hot_race['Race_Hispanic']],
                            'Race_Other': [one_hot_race['Race_Other']],
                            'Race_Two-or-more': [one_hot_race['Race_Two-or-more']],
                            'Race_White': [one_hot_race['Race_White']],
                            'ParentalEducation_Bachelors+': [one_hot_parenteducation['ParentalEducation_Bachelors+']],
                            'ParentalEducation_HS': [one_hot_parenteducation['ParentalEducation_HS']],
                            'ParentalEducation_SomeCollege': [one_hot_parenteducation['ParentalEducation_SomeCollege']],
                            'Locale_Rural': [one_hot_local['Locale_Rural']],
                            'Locale_Suburban':[one_hot_local['Locale_Suburban']],
                            'Locale_Town': [one_hot_local['Locale_Town']] }, index=[0])
    
    
    input_data = input_data[feature_order]
    
    # Prediction 
    try:
        gpa = model.predict(input_data)[0]
    
        st.subheader("📊 Predicted GPA")
        st.metric(label="Estimated GPA", value=f"{gpa:.2f}")

        if gpa >= 3.7:
            st.success(f"🎓 Outstanding performance{name and f', {name}' or ''}! You're excelling academically — keep aiming high!")
            st.balloons()
        elif gpa >= 3.3:
            st.info(f"🌟 Great job{name and f', {name}' or ''}! You're doing very well. Stay consistent and success will follow.")
        elif gpa >= 2.5:
            st.info(f"👍 Good effort{name and f', {name}' or ''}. There's solid progress here — a little more dedication can take you further.")
        elif gpa >= 2.0:
            st.warning(f"⚠️ Your GPA is currently average{name and f', {name}' or ''}. Consider refining your study strategy and seeking support if needed.")
        else:
            st.error(f"🚨 GPA is below average{name and f', {name}' or ''}. Don’t be discouraged — reflect, reset, and reach out for guidance. You’ve got this!")
    except Exception as e:
        st.error(f"Prediction failed: {e}")