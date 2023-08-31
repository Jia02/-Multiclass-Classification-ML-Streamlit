# import all the app dependencies
import pandas as pd
import numpy as np
import sklearn
import streamlit as st
import joblib
from IPython import get_ipython
from PIL import Image

# load the encoder and model object
model = joblib.load("rta_model_deploy3.joblib")
encoder = joblib.load("ordinal_encoder2.joblib") 

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_title=" Prediction of Vehicle Accident Severity",
        page_icon="🚗", layout="wide")

#creating option list for dropdown menu
options_day = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
options_age = ['Under 18', '18-30', '31-50', 'Over 51', 'Unknown']

# number of vehicle involved: range of 1 to 7
# number of casualties: range of 1 to 8
# hour of the day: range of 0 to 23

options_types_collision = [
    'Vehicle with vehicle collision',
    'Collision with roadside objects',
    'Collision with pedestrians',
    'Rollover',
    'Collision with animals',
    'Collision with roadside-parked vehicles',
    'Fall from vehicles',
    'With Train',
    'Unknown',
    'Other',]

options_sex = ['Male','Female','Unknown']

options_education_level = ['Junior high school','Elementary school','High school',
              'Unknown','Above high school','Writing & reading','Illiterate']

options_services_year = ['Below 1yr','1-2yr','2-5yrs','5-10yrs','Above 10yr','Unknown']

options_acc_area = ['Office areas', 'Residential areas', ' Church areas',
    ' Industrial areas', 'School areas', ' Recreational areas',
    ' Outside rural areas', ' Hospital areas', ' Market areas',
    'Rural village areas', 'Unknown', 'Rural village areasOffice areas',
    'Recreational areas', 'Other']

# features list
features = ['Number_of_vehicles_involved','Number_of_casualties','Hour_of_Day','Type_of_collision','Age_band_of_driver','Sex_of_driver',
    'Educational_level','Service_year_of_vehicle','Day_of_week','Area_accident_occured']

# define a main() function to take inputs from user in form based approach
def main():
    with st.form("road_traffic_severity_form"):
       st.subheader("Please enter the following inputs:")
       Day_week = st.selectbox("Day of the week:", options=options_day)
       Hour = st.slider("Hour of the day:", 0, 23, value=0, format="%d")
       collision = st.selectbox("Type of collision:",options=options_types_collision)
       No_vehicles = st.slider("Number of vehicles involved:",1,7, value=0, format="%d")
       No_casualties = st.slider("Number of casualties:",1,8, value=0, format="%d")
       Accident_area = st.selectbox("Area of accident:", options=options_acc_area)
       Age_band = st.selectbox("Driver age group?:", options=options_age)
       Sex = st.selectbox("Sex of the driver:", options=options_sex)
       Education = st.selectbox("Education of driver:",options=options_education_level)
       service_vehicle = st.selectbox("Service year of vehicle:", options=options_services_year)
    
       submit = st.form_submit_button(label="Predict")
       st.write('<div style="display: flex; justify-content: center;">', submit, '</div>', unsafe_allow_html=True)

# encode using ordinal encoder and predict
    if submit:
        input_array = np.array([collision,
                    Age_band,Sex,Education,service_vehicle,
                    Day_week,Accident_area], ndmin=2)
            
        encoded_arr = list(encoder.transform(input_array).ravel())
            
        num_arr = [No_vehicles,No_casualties,Hour]
        pred_arr = np.array(num_arr + encoded_arr).reshape(1,-1) 
      
# predict the target from all the input features
# 0: Fatal Injury, 1: Major Injury, 2: Minor Injury
        prediction = model.predict(pred_arr)
        
        if prediction == 0:
            text="The severity prediction is ❗ fatal injury ❗"

        elif prediction == 1:
            text="The severity prediction is major injury ⚠"
        else:
            text="The severity prediction is minor injury"

        st.markdown(f'<p style="text-align:center; font-size:26px; font-weight:bold;">{text}</p>', unsafe_allow_html=True)

    
    st.write("Developed By: YJ Lim")
    st.markdown("""
    [LinkedIn](https://www.linkedin.com/in/yujia-lim-b85081213/) | [Github Repository](https://github.com/Jia02/-Multiclass-Classification-ML-Streamlit)
    """)

a,b,c = st.columns([0.2,0.6,0.2])
with b:
 st.image("banner-pic.jpeg", use_column_width=True)

# Title to web app using html syntax
st.markdown("<h1 style='text-align: center;'> App for Predicting Accident Severity  🚗🤕 </h1>", unsafe_allow_html=True)

# description about the project and code files       
st.subheader("📝Description:")
st.text("""
        This [dataset](https://www.narcis.nl/dataset/RecordID/oai%3Aeasy.dans.knaw.nl%3Aeasy-dataset%3A191591) originates from the police departments of Addis Ababa Sub-city and serves as the foundation for master's research endeavors. 
        The dataset's compilation draws from manual records documenting road traffic accidents spanning 2017 to 2020, encompassing 32 distinct features and 12,316 instances of accidents.  
        This application aims to predict the severity of road traffic accidents based on 10 differnet features modified by the user in order to discern significant accident causes through comprehensive analysis.

""")

st.subheader("❓Statement of Problem:")
st.text("""The target feature is `Accident_severity` which is a multi-class data. 
The task aims to classify this variable based on the other 31 features.
The metric for evaluation is f1-score.
Random forest classification model is used to predict the target feature based on the selected 10 input feature.""")
  
# run the main function        
if __name__ == '__main__':
  main()
          
