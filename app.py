from operator import index
from typing import Container
from unittest import result
from pandas import wide_to_long
import streamlit as st
from sklearn import svm
import pandas as pd
import numpy as np
import seaborn as sb
from model import Model

st.set_page_config(page_title="Depression Indicator", page_icon=":umbrella:", layout="wide")

def main():
        user()
    
def user():
    # ---- Header ----
    dh1, dh2, dh3, dh4, dh5= st.columns(5)
    hd1, hd2, hd3= st.columns(3)

    st.info('')

    dh3.image("2622142.png", width=200)
    hd2.title("Depression Indicator")
    hd2.write("")
    hd2.write("")
    
    #---- Questions ----
    col1, col2, col3, col4, col5 = st.columns(5)
    ans = {'Not at all' : 0, 'Several days' : 1, 'More than half the days' : 2, 'Nearly every day' : 3}
    q1 = col1.radio("Little interest or pleasure in doing things", ans, key="a1")
    global a1
    if q1 == 'Not at all':
        a1 = 0
    if q1 == 'Several days':
        a1 = 1
    if q1 == 'More than half the days':
        a1 = 2
    if q1 == 'Nearly every day':
        a1 = 3  

    col1.write("")
    col1.write("")
    col1.write("")

    q2 = col1.radio("Feeling down, depressed, or hopeless",
    ('Not at all', 'Several days', 'More than half the days', 'Nearly every day'))
    global a2
    if q2 == 'Not at all':
        a2 = 0
    if q2 == 'Several days':
        a2 = 1
    if q2 == 'More than half the days':
        a2 = 2
    if q2 == 'Nearly every day':
        a2 = 3  

    q3 = col2.radio("Trouble falling or staying asleep, or sleeping too much",
    ('Not at all', 'Several days', 'More than half the days', 'Nearly every day'))
    global a3
    if q3 == 'Not at all':
        a3 = 0
    if q3 == 'Several days':
        a3 = 1
    if q3 == 'More than half the days':
        a3 = 2
    if q3 == 'Nearly every day':
        a3 = 3  

    col2.write("")
    col2.write("")
    

    q4 = col2.radio("Feeling tired or having little energy",
    ('Not at all', 'Several days', 'More than half the days', 'Nearly every day'))
    global a4
    if q4 == 'Not at all':
        a4 = 0
    if q4 == 'Several days':
        a4 = 1
    if q4 == 'More than half the days':
        a4 = 2
    if q4 == 'Nearly every day':
        a4 = 3  

    q5 = col3.radio("Poor appetite or overeating",
    ('Not at all', 'Several days', 'More than half the days', 'Nearly every day'))
    global a5
    if q5 == 'Not at all':
        a5 = 0
    if q5 == 'Several days':
        a5 = 1
    if q5 == 'More than half the days':
        a5 = 2
    if q5 == 'Nearly every day':
        a5 = 3  

    col3.write("")
    col3.write("")
    col3.write("")

    q6 = col3.radio("Feeling bad about yourself, or that you are a failure or have let yourself or your family down",
    ('Not at all', 'Several days', 'More than half the days', 'Nearly every day'))
    global a6
    if q6 == 'Not at all':
        a6 = 0
    if q6 == 'Several days':
        a6 = 1
    if q6 == 'More than half the days':
        a6 = 2
    if q6 == 'Nearly every day':
        a6 = 3  

    q7 = col4.radio("Trouble concentrating on things, such as reading the newspaper/watching television",
    ('Not at all', 'Several days', 'More than half the days', 'Nearly every day'))
    global a7
    if q7 == 'Not at all':
        a7 = 0
    if q7 == 'Several days':
        a7 = 1
    if q7 == 'More than half the days':
        a7 = 2
    if q7 == 'Nearly every day':
        a7 = 3  

    col4.write("")
    col4.write("")

    q8 = col4.radio("Moving or speaking so slowly that other people could have noticed. Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual",
    ('Not at all', 'Several days', 'More than half the days', 'Nearly every day'))
    global a8
    if q8 == 'Not at all':
        a8 = 0
    if q8 == 'Several days':
        a8 = 1
    if q8 == 'More than half the days':
        a8 = 2
    if q8 == 'Nearly every day':
        a8 = 3  

    q9 = col5.radio("Thoughts that you would be better off dead, or of hurting yourself",
    ('Not at all', 'Several days', 'More than half the days', 'Nearly every day'))
    global a9
    if q9 == 'Not at all':
        a9 = 0
    if q9 == 'Several days':
        a9 = 1
    if q9 == 'More than half the days':
        a9 = 2
    if q9 == 'Nearly every day':
        a9 = 3  

    col5.write("")
    col5.write("")

    q10 = col5.radio("If you've had any of the issues above, how difficult have these problems made it for you to do your work, take care of things at home, or get along with other people?",
    ('Not difficult at all', 'Somewhat difficult', 'Very difficult', 'Extremely difficult'))
    global a10
    if q10 == 'Not difficult at all':
        a10 = 0
    if q10 == 'Somewhat difficult':
        a10 = 1
    if q10 == 'Very difficult':
        a10 = 2
    if q10 == 'Extremely difficult':
        a10 = 3  

    col3.write("")
    col3.write("")
    col3.write("")

    st.info('')
    st.write("")
    st.write("")

    f1, f2, f3, f4, f5, f6, f7, f8, f9= st.columns(9)
    e1, e2, e3= st.columns(3)
    
    # ----Result button----
    submit = f5.button("See Result")

    e2.write("")
    e2.write("")

    if submit:
        e2.warning(predict())

    st.write("")
    st.write("")
    st.write("")

    # ----Footer----
    my_expander = st.expander(label='WIA1006 Machine Learning project by')
    with my_expander:
        'IDK Machine Learning'

def predict():
        # Using the trained SVM classifier to predict the result of user's input
        model = Model()
        input = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10]
        classifier = model.SVMclassifier()
        prediction = classifier.predict([input])
        if prediction[0] == 0:
                result = 'Prediction: No Depression'
        if prediction[0] == 1:
                result = 'Prediction: Mild Depression'
        if prediction[0] == 2:
                result = 'Prediction: Moderate Depression'
        if prediction[0] == 3:
                result = 'Prediction: Moderately Severe Depression'
        if prediction[0] == 4:
                result = 'Prediction: Severe Depression'
        return result

if __name__ == "__main__":
    main()