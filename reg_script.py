import pandas as pd
import plotly.express as px
from IPython.display import Image
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import statsmodels.api as sm
df = pd.read_csv('cars_500k.csv')
dfSample = df.sample(250000)
dfSample.to_csv('cars_250k.csv')
def reg_sp():

    dfSample = df.sample(1000)
    reg_fig = px.scatter(dfSample, x="Year", y='Price', trendline="ols")
    return reg_fig

def make_mileage (make):
    mileage_threshold = df.groupby(make).filter(lambda x: len(x) > 10000)
    mileage_fig = px.histogram(mileage_threshold, x=make, y="Mileage", histfunc='avg')
    mileage_fig.show()

app = dash.Dash()
app.layout = html.Div([
    dcc.Dropdown(
        id='product',
        options=[
            {'label': 'Make', 'value': 'Make'},
            {'label': 'Model', 'value': 'Model'}], value='Make'),
    dcc.Graph(id = 'bar_count', figure=make_mileage("Make"))
])

@app.callback(
    Output(component_id='bar_count', component_property='figure'),
    [
     Input(component_id='product', component_property='value')])

def update_figure1(product):
    count_fig2 = make_mileage(product)

    return count_fig2

app.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter