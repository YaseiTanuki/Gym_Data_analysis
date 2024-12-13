import pandas as pd
import streamlit as st
from pickle import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import cross_val_score
import plotly.express as px

from method import *
import altair as alt
plt.style.use('dark_background')

st.set_page_config(
    page_title="Gym Dataset",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")
df = pd.read_csv('data.csv')

with st.sidebar:
    st.title('Overview')
    observe_variable = st.selectbox('Select variable to observe', df.columns)

    st.markdown('---')
    
    st.title('Correlation')
    st.header('Correlation')
    numeric_df = df.select_dtypes(include='number')
    corrVars = st.multiselect('Select variable for correlation check', numeric_df.columns, default=numeric_df.columns)
    st.header('Scatter')
    scatterX = st.selectbox('Select X', df.columns)
    scatterY = st.selectbox('Select Y:', df.columns)

    st.markdown('---')

    st.title('Model')
    modelVars = st.multiselect('Select variables', df.columns, default=['Age'])
    targetVariable = st.selectbox('Select target variable', df.columns)
    if (modelVars) and (targetVariable):
        X = df[modelVars]
        y = df[targetVariable]
        le = LabelEncoder()
        X = pd.get_dummies(X)
        if y.dtype == 'object':
            y = le.fit_transform(y)



col_first = st.columns((0.6, 0.3))
with col_first[0]:
    st.title('Dataset summary')
    st.dataframe(summary(df))

st.title('Data Valuation')
bins = [0, 18.5, 24.9, 29.9, np.inf]
labels = ['Underweight', 'Normal', 'Overweight', 'Obesity']
df['BMI_Category'] = pd.cut(df['BMI'], bins=bins, labels=labels)

col_value = st.columns(3)
with col_value[0]:
    fig, ax = plt.subplots()
    ax = sns.boxplot(x='Workout_Type', y='Calories_Burned', data=df, palette='Set3')
    plt.title('Calories Burned by Workout Type')
    plt.xlabel('Workout Type')
    plt.ylabel('Calories Burned')
    with st.container(border=True):
        st.pyplot(fig)

with col_value[1]:
    age_bins = [18, 25, 35, 45, 55, 65]  # Các khoảng độ tuổi
    age_labels = ['26-35', '36-45', '46-55', '56-65', '66+']  # Tên các nhóm độ tuổi
    df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
    plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots()
    ax = sns.barplot(x='Age_Group', y='Calories_Burned', hue='Workout_Type', data=df, palette='viridis')
    plt.title('Calories Burned vs Age Group by Workout Type')
    plt.xlabel('Age Group')
    plt.ylabel('Average Calories Burned')
    plt.legend(title='Workout Type')
    with st.container(border=True):
        st.pyplot(fig)


with col_value[2]:

    fig, ax = plt.subplots()
    ax = sns.countplot(x='BMI_Category', hue='Workout_Type', data=df, palette='Set2')
    plt.title('BMI Categories by Workout Type')
    plt.xlabel('BMI Category')
    plt.ylabel('Count')
    plt.legend(title='Workout Type')
    with st.container(border=True):
        st.pyplot(fig)

st.title('Data observation')
col_observe = st.columns(3)
bar, pie, box = univariateAnalysis_category(observe_variable, df)
with col_observe[0]:
    with st.container(border=True):
        st.plotly_chart(bar)

with col_observe[1]:
    with st.container(border=True):
        st.plotly_chart(pie)

with col_observe[2]:
    with st.container(border=True):
        st.plotly_chart(box)


st.markdown("""---""")
col_second = st.columns((0.7, 0.3), gap='medium')
with col_second[0]:
    st.header('Correlation')
    corrMap = correlationAnalysis_category(df[corrVars])
    st.plotly_chart(corrMap)

with col_second[1]:
    st.header('Regression Plot')
    fig, ax = plt.subplots()
    ax = sns.regplot(df, x=scatterX, y=scatterY)
    st.pyplot(fig)

st.markdown("""---""")

st.title('Create Model')
model = None
col_thrid = st.columns((0.3, 0.4, 0.3), gap='medium')

with col_thrid[0]:
    st.header('Test set')
    test_size = st.slider('Test size', min_value=0.1, max_value=1.0, step=0.05, value=0.2)
    random_state = st.number_input('Random state', value=1, min_value=1, max_value=4294967295)
    if (modelVars) and (targetVariable):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    st.header('Select type of model')
    modelTypes = ['Linear Regression', 'Polynominal Regression', 'Decision Tree Classifier']
    modelType = st.selectbox('Select type of model', modelTypes)
    if modelType == 'Polynominal Regression':
        degree = st.slider('Degree', min_value=1, max_value=5, step=1, value=2)
    if (st.button('Train')):
        if modelType == 'Polynominal Regression':
            pr = PolynomialFeatures(degree)
            x_train = pr.fit_transform(X_train)
            x_test = pr.fit_transform(X_test)
            model = trainLinear(X_train, y_train)
        if modelType == 'Decision Tree Classifier':
            model = trainDecisionTree(X_train, y_train)
            st.text(model.predict(X_test))
        else:
            model = trainLinear(X_train, y_train)

with col_thrid[1]:
        st.header('Evaluate model')
        if model:
            MSE, r_square = evaluate(model, X_test, y_test)
            st.text(f'MSE: {MSE}')
            st.text(f'R Square: {r_square}')
            fig = actualPredictShow(y_test, model.predict(X_test))
            st.pyplot(fig)

st.markdown("""---""")
st.title('Model detail')
col_detai = st.columns(3)
with col_detai[0]:
    if model:
        with st.container(border=True):
            cross_score = cross_val_score(model, X, y, cv=5)
            st.bar_chart(cross_score)
with col_detai[1]:
    if modelType == 'Polynominal Regression':
        mses = []
        r_squares = []
        for n in range(1, 6):
            test_model = trainLinear(X_train, y_train)
            mse, r_square = evaluate(test_model, X_test, y_test)
            mses.append(mse)
            r_squares.append(r_square)
        chart_data = pd.DataFrame({'MSE': mses, 'R Square': r_squares})
        chart_data.index += 1
        st.dataframe(chart_data)
        st.line_chart(chart_data)



