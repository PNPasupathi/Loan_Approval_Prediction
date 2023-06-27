import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64


# Background Image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
        unsafe_allow_html=True
    )
add_bg_from_local('Images/bg1.jpg')

scaler=pickle.load(open('scaled_feature.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title('Loan Status Prediction')
st.markdown('#')
gender=st.selectbox('Select the Gender',['Male','Female'])
married=st.selectbox('Are you Married or Not',['Yes','NO'])
dependence=st.selectbox('Enter the Number of Dependence',['0','1','2','3+'])
education=st.selectbox('Enter you Education',['Graduate','Not Graduate'])
selfemp=st.selectbox('Are you Self Employee',['Yes','No'])
applicantincome=st.number_input('Enter Applicant Income',min_value=0)
coapplicantincome=st.number_input('Enter Coapplicant Income',min_value=0)
loanamount=st.number_input('Enter the Loan Amount',min_value=0)
loanamountterm=st.number_input('Enter the Loan Amount Term',min_value=0)
credithistory=st.number_input('Enter the Credit History',min_value=0)
propertyarea=st.selectbox('Enter the Property Area',['Urban','Rural','Semiurban'])
st.markdown('#')
st.write('')
if gender=='Male':
    gender=1
else:
    gender=0
if married=='Yes':
    married=1
else:
    married=0
if dependence=='0':
    dependence=0
elif dependence=='1':
    dependence=1
elif dependence=='2':
    dependence=2
else:
    dependence=3
if education=='Graduate':
    education=1
else:
    education=0
if selfemp=='Yes':
    selfemp=1
else:
    selfemp=0
if propertyarea=='Semiurban':
    propertyarea=0
elif propertyarea=='Urban':
    propertyarea=1
else:
    propertyarea=2
btn1,btn2,btn3=st.columns([2,1,2])
with btn2:
    predictbtn=st.button('Predict')
if predictbtn==True:
    scaled=scaler.transform(np.array([gender,married,dependence,education,selfemp,applicantincome,coapplicantincome,loanamount,loanamountterm,credithistory,propertyarea]).reshape(1,-1))
    result=model.predict(np.array(scaled))
    res1,res2,res3=st.columns([1,2,1])
    with res2:
        if result[0]==0:
            st.markdown('#')
            st.image('Images/loanrej.png')
        else:
            st.markdown('#')
            st.image('Images/loanapp.png')
