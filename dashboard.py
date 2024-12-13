import streamlit as st
from pickle import dump
from sklearn.model_selection import train_test_split

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
    variable = st.selectbox('Select aim variable', df.columns)
    observe_variable = st.selectbox('Select variable to observe', df.columns)

    st.markdown('---')
    
    st.title('Correlation')
    st.header('Correlation')
    numeric_df = df.select_dtypes(include='number')
    corrVars = st.multiselect('Select variable for correlation check', numeric_df.columns)
    st.header('Scatter')
    scatterX = st.selectbox('Select X', df.columns)
    scatterY = st.selectbox('Select Y:', df.columns)

col_first = st.columns((0.6, 0.3))
with col_first[0]:
    st.title('Dataset summary')
    st.dataframe(summary(df))

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
    corrMap = correlationAnalysis_category(df[corrVars])
    st.plotly_chart(corrMap)

with col_second[1]:
    st.header('Select 2 variables for scatter plot')
    fig, ax = plt.subplots()
    ax = sns.regplot(df, x=scatterX, y=scatterY)
    st.pyplot(fig)

st.markdown("""---""")

st.title('Create Model')

col_thrid = st.columns((0.2, 0.3, 0.5), gap='medium')
trained_model_count = 0
with col_thrid[0]:
    st.header('Select variable')
    option = []
    for col in df.columns:
        op = st.checkbox(col, False, key=col+' for model')
        if op:
            option.append(col)
    X = df[option]
    y = df[variable]

with col_thrid[1]:
    st.header('Test size')
    test_size = st.slider('Test size', min_value=0.1, max_value=1.0, step=0.05, value=0.2)
    random_state = st.number_input('Random state', value=1, min_value=1, max_value=4294967295)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    st.header('Select type of model')
    name = st.text_input('Name the model:')
    modelTypes = ['Linear Regression', 'Polynominal Regression']
    modelType = st.selectbox('Select type of model', modelTypes)
    if modelType == 'Polynominal Regression':
        degree = st.slider('Degree', min_value=1, max_value=5, step=1, value=2)
    if (st.button('Train')):
        if modelType == 'Polynominal Regression':
            model = trainPolynomial(X_train, y_train, degree)
        else:
            model = trainLinear(X_train, y_train)

        trained_model_count += 1
        output = open(f'models/{name}.pkl', 'wb')
        dump(model, output)
        output.close()

with col_thrid[2]:
        st.header('Evaluate model')
        if (trained_model_count > 0):
            MSE, r_square = evaluate(model, X_test, y_test)
            st.subheader(f'MSE: {MSE}')
            st.subheader(f'r_square: {r_square}')
            fig = actualPredictShow(y_test, model.predict(X_test))
            st.pyplot(fig)