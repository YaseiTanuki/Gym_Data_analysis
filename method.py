import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import seaborn as sns



def summary(df):
    summ = pd.DataFrame(df.dtypes, columns=['data type'])
    summ['#missing'] = df.isnull().sum().values
    summ['Duplicate'] = df.duplicated().sum()
    summ['#unique'] = df.nunique().values
    desc = pd.DataFrame(df.describe(include='all').transpose())
    summ['min'] = desc['min'].values
    summ['max'] = desc['max'].values
    summ['avg'] = desc['mean'].values
    summ['std dev'] = desc['std'].values
    summ['top value'] = desc['top'].values
    summ['Freq'] = desc['freq'].values

    return summ

def univariateAnalysis_category(cols, df=None):
    print("Distribution of", cols)
    print("_" * 60)
    colors = [
        '#FFD700', '#FF6347', '#40E0D0', '#FF69B4', '#7FFFD4',
        '#FFA500', '#00FA9A', '#FF4500', '#4682B4', '#DA70D6',
        '#FFB6C1', '#FF1493', '#FF8C00', '#98FB98', '#9370DB',
        '#32CD32', '#00CED1', '#1E90FF', '#FFFF00', '#7CFC00'
    ]
    value_counts = df[cols].value_counts()

    # Create bar plot
    fig = px.bar(
        value_counts,
        x=value_counts.index,
        y=value_counts.values,
        title=f'{cols}',
        labels={'x': 'Categories', 'y': 'Count'},
        color_discrete_sequence=[colors]
    )
    fig.update_layout(
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font=dict(color='white', size=12),
        title_font=dict(size=30),
        legend_font=dict(color='white', size=12),
        width=500,  # Adjusted width
        height=400  # Adjusted height
    )

    # Calculate percentage
    percentage = (value_counts / value_counts.sum()) * 100

    # Create pie chart
    pie = px.pie(
        values=percentage,
        names=value_counts.index,
        title=f'{cols}',
        labels={'names': 'Categories', 'values': 'Percentage'},
        hole=0.5,
        color_discrete_sequence=colors
    )
    pie.add_annotation(
        x=0.5, y=0.5,
        text=f'{cols}',
        font=dict(size=18, color='white'),
        showarrow=False
    )
    pie.update_layout(
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font=dict(color='white', size=12),
        title_font=dict(size=30),
        legend=dict(x=0.9, y=0.5),
        legend_font=dict(color='white', size=12),
        width=500,  # Adjusted width
        height=400  # Adjusted height
    )

    box = px.box(df, y=cols)
    box.update_layout(
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font=dict(color='white', size=12),
        legend_font=dict(color='white', size=12),
        width=500,  # Adjusted width
        height=400  # Adjusted height
    )

    return fig, pie, box

def correlationAnalysis_category(df):
    correlation_matrix = df.corr()
    fig = go.Figure(data=go.Heatmap(z=correlation_matrix, x=correlation_matrix.columns, y=correlation_matrix.columns))
    fig.update_layout(title='Correlation Heatmap')
    return fig

def trainLinear(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def trainPolynomial(X, y, degree):
    input = [('scale', StandardScaler()), ('poly', PolynomialFeatures(degree)), ('model', LinearRegression())]
    model = Pipeline(input)
    model.fit(X, y)
    return model

def evaluate(model, X, y):
    MSE = mean_squared_error(y, model.predict(X))
    r_squared = model.score(X, y)
    return MSE, r_squared

def actualPredictShow(y_true, y_predict):
    fig, ax = plt.subplots()
    ax = sns.distplot(y_true, color='blue', hist=False, label='Actual')
    sns.distplot(y_predict, color='red', hist=False, label='Predicted')
    plt.title('Actual vs Predicted')
    plt.legend()
    return fig