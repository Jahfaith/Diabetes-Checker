#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing relevant libraries.

import streamlit as st
import pickle
import pandas as pd
import numpy as np


# In[2]:


# loading the model.

model = pickle.load(open('log2_model.pkl', 'rb'))


# In[5]:


# Heading.
st.write('Diabetes Checker')

# Input from the user.
Age = st.text_input("How old are yo?")
Glucose = st.text_input("Please enter your Glucose level")
BMI = st.text_input("What is your BMI?")
BloodPressure = st.text_input("What is your Blood Pressure level?")
Insulin = st.text_input("Please enter your Insulin level")
Pregnancies = st.text_input("How many pregnacies have yo had?")

# Making prediction.
if st.button("Check Now"):
    # Arranging the features in the order in which they were fitted/trained with the Log2_model.
    features = [[Glucose, BMI, Age, BloodPressure, Insulin, Pregnancies]]
    feat = {'Glucose': (Glucose), 'BMI': (BMI), 'Age': (Age), 'BloodPressure': (BloodPressure), 'Insulin': (Insulin), 'Pregnancies': (Pregnancies)}
    df = pd.DataFrame([list(feat.values())], columns=['Glucose', 'BMI', 'Age', 'BloodPressure', 'Insulin', 'Pregnancies'])
    prediction = model.predict(df)

    output = int(prediction[0])
    if output == 0:
        text = 'Yo most likely DO NOT have Diabetes.\nKindly visit your doctor for more checks.'
    else:
        text = 'Yo most likely HAVE Diabetes.\nPlease see a doctor for confirmation and possibly treatment.'
        
    st.success(text)
    


# In[ ]:





# In[ ]:





# In[ ]:




