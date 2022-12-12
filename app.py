import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

st.title("Let's see if my algorithm can predict whether or not you use LinkedIn")
st.text("Made by Michael W. Rowe II")

#Age
age = st.slider(label="Enter you age", min_value=1, max_value=98,value=7)
#st.write("Your age is: ", age)


#Gender

gender = st.selectbox("What is your gender?", options = ["","Male", "Female"] )
if gender == "Female":
           gender = 1
else:
           gender = 0
           
#st.write(gender)

#Married

married = st.selectbox("Are you married?", options = ["","Yes", "No"] )
if married == "Yes":
           married = 1
else:
           married = 0

#st.write(married)

#Parent
parent = st.selectbox("Are you a parent?", options = ["","Yes", "No"] )
if parent == "Yes":
           parent = 1
else:
           parent = 0

#st.write(parent)

#Highest Level of Eduction
options = ("",
           "Less than high school (Grades 1-8 or no formal schooling)",
           "High school graduate (Grade 12 with diploma or GED certificate)",
           "High school graduate (Grade 12 with diploma or GED certificate)",
           "Some college, no degree (includes some community college)",
           "Two-year associate degree from a college or university",
           "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)",
           "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)",
           "Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)")

education = st.selectbox("Highest Level of Education", options = range(len(options)), format_func=lambda x: options[x])

#st.write(education)


##Income
options = ("","Less than $10,000",
                        "10 to under $20,000",
                        "20 to under $30,000",
                        "30 to under $40,000",
                        "40 to under $50,000",
                        "50 to under $75,000",
                        "75 to under $100,000",
                        "100 to under $150,000",
                        "$150,000 or more")

income = st.selectbox("Income Level", options = range(len(options))              
              , format_func=lambda x: options[x])


#st.write(income)

#if income != "" && education != "" && parent != "" && married != "" && gender != "" && age != "":
def clean_sm(x):
        x = np.where(x == 1, 1, 0)
        return(x)           
s = pd.read_csv("social_media_usage.csv")

ss = pd.DataFrame({
           "sm_li":clean_sm(s["web1h"]),
           "income": np.where(s["income"] >9, np.nan, s["income"]),
           "education": np.where(s["educ2"]>8, np.nan, s["educ2"]),
           "parent": clean_sm(s["par"]),
           "married": clean_sm(s["marital"]),
           "female": clean_sm(s["gender"]),
           "age": np.where(s["age"]> 98, np.nan, s["age"])})

ss= ss.dropna()
y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female", "age"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.2, random_state = 777)
lr = LogisticRegression(class_weight = "balanced")
lr.fit(X_train, y_train) 
observation = [income, education, parent, married, gender, age]

with st.form("key1"):
           submit = st.form_submit_button("Load Prediction")
           if submit:
                      pred_outcome = lr.predict([observation])
                      pred_outcome_probability = lr.predict_proba([observation])
                      if pred_outcome == [1]:
                                 st.write("You are a LinkedIn User")
                      else:
                                 st.write("You are not a LinkedIn User")
                      st.text("The probability you're a LinkedIn user is:" pred_outcome_probability)
                      st.text(pred_outcome)

