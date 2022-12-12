import streamlit as st
import numpy as np
import pandas as pd
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix

st.title("Let's see if I can predict if you used LinkedIn")

#Age
age = st.slider(label="Enter you age", min_value=1, max_value=98,value=7)
#st.write("Your age is: ", age)


#Gender
gender = st.selectbox("What is your gender?", options = ["","Male", "Female"])

#Married
married = st.selectbox("Are you married?", options = ["","Yes", "No"])

#Parent
parent = st.selectbox("Are you a parent?", options = ["","Yes", "No"])

#Highest Level of Eduction
education = st.selectbox("Highest Level of Education", 
              options = ["",
                        "Less than high school (Grades 1-8 or no formal schooling)",
                        "High school graduate (Grade 12 with diploma or GED certificate)",
                        "High school graduate (Grade 12 with diploma or GED certificate)",
                        "Some college, no degree (includes some community college)",
                        "Two-year associate degree from a college or university",
                        "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)",
                        "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)",
                        "Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)"])


##Income
income = st.selectbox("Income Level",                
              options = ["Less than $10,000",
                        "10 to under $20,000",
                        "20 to under $30,000",
                        "30 to under $40,000",
                        "40 to under $50,000",
                        "50 to under $75,000",
                        "75 to under $100,000",
                        "100 to under $150,000",
                        "$150,000 or more",
                        ""],
                        index = 10)

#If income != "" & education != "" & parent != "" married != "" 
  #s = pd.read_csv("social_media_usage.csv")
  #s
