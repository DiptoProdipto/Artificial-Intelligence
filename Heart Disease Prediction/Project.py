import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from PIL import Image
import streamlit as st

st.write("""# Heart Disease Prediction""")

data = pd.read_csv('C:/Users/Dipto Prodipto/Desktop/cardio_train.csv',sep=';')

data['years'] = (data['age']/365).round(0)
data['years'] = pd.to_numeric(data['years'],downcast='integer')

x = data.iloc[:,[2,3,4,5,6,7,8,9,10,11,13]]
y = data.iloc[:,12]

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.20,random_state=1)

#sc = StandardScaler()
#xtrain = sc.fit_transform(xtrain)
#xtest = sc.transform(xtest)

def get_user_input():
        gender = st.sidebar.slider('gender',1,2,2)
        height = st.sidebar.slider('height',55,250,156)
        weight = st.sidebar.slider('weight',10,200,55)
        years = st.sidebar.slider('years',10,100,41)
        ap_hi = st.sidebar.slider('ap_hi',-150,16020,140)
        ap_lo = st.sidebar.slider('ap_lo',-70,11000,90)
        cholesterol = st.sidebar.slider('cholesterol',1,3,2)
        gluc = st.sidebar.slider('gluc',1,3,1)
        smoke = st.sidebar.slider('smoke',0,1,0)
        alco = st.sidebar.slider('alco',0,1,0)
        active = st.sidebar.slider('active',0,1,1)





        user_data = {'gender':gender,
                     'height':height,
                     'weight':weight,
                     'years':years,
                     'ap_hi':ap_hi,
                     'ap_lo':ap_lo,
                     'cholesterol':cholesterol,
                     'gluc':gluc,
                     'smoke':smoke,
                     'alco':alco,
                     'active':active,
                     }

        features = pd.DataFrame(user_data, index = [0])
        return features

user_input = get_user_input()
st.subheader('User Input:')
st.write(user_input)

rfc = RandomForestClassifier(n_estimators=100, criterion = 'entropy', random_state=1)
rfc.fit(xtrain,ytrain)

st.subheader('Model Test Accuracy Score: ')
st.write(str(accuracy_score(ytest,rfc.predict(xtest))*100)+'%')

prediction = rfc.predict(user_input)
st.subheader('Classification: ')
st.write(prediction)

