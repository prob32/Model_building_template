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


df["Model"] = df["Make"] + " " + df["Model"]

scatterplot_items=['Price','Mileage','Year']

def categorical_coding(column_name,dataframe):
    dataframe[column_name] = dataframe[column_name].astype('category')
    dataframe[column_name] = dataframe[column_name].cat.codes
def counts_viz (type):
    make_more_than_twenty_thousand = df.groupby(type).filter(lambda x: len(x) > 5000)
    count_fig = px.histogram(make_more_than_twenty_thousand,x=type)
    return count_fig

def reg_sp(X,Y):
    dfSample = df.sample(1000)
    reg_fig = px.scatter(dfSample, x=X, y=Y, trendline="ols")
    return reg_fig

def make_mileage (make):
    mileage_threshold = df.groupby(make).filter(lambda x: len(x) > 10000)
    mileage_fig = px.histogram(mileage_threshold, x=make, y="Mileage", histfunc='avg')
    mileage_fig.show()


myheading1='Carz Test'
colors = {
    'background': '#111111',
    'text': 'rgb(60,5,224)'
}
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1(myheading1),


    ### Choose make or model
    html.Label('Count by Make or Model'),
    dcc.Dropdown(
        id='product',
        options=[
            {'label':'Make','value': 'Make'},
            {'label':'Model','value': 'Model'}],value = 'Make'),


    ### Graph-barcount
    dcc.Graph(id = 'bar_count', figure=counts_viz('Make')),



    ### Scatterplotter-inputs
    ## X input
    html.Label('Input X variable'),
    dcc.Dropdown(
        id='X',
        options=[{'label': i, 'value': i} for i in scatterplot_items], value='Mileage'),

    ### Y Input
    html.Label('Input Y variable'),
    dcc.Dropdown(
        id='Y',
        options=[{'label': i, 'value': i} for i in scatterplot_items], value='Price'),

    ####Scatterplotter-outputs
    dcc.Graph(id = 'scat_plot',figure=reg_sp('Mileage','Price')),
    ### Output function
    html.H4(id='div3', style={'color': colors['text']}),], style = {'columnCount': 2,'width': '100%',  'vertical-align': 'middle', 'display': 'inline-block'})

@app.callback(
    [Output(component_id='bar_count', component_property='figure'),

     Output(component_id='scat_plot', component_property='figure')],

    [
     Input(component_id='product', component_property='value'),
     Input(component_id='X', component_property='value'),
     Input(component_id='Y', component_property='value')
     ])



def update_figure(product,X,Y):
       count_fig2 = counts_viz(product)
       mileage_fig = make_mileage(product)
       sp_fig = reg_sp(X,Y)
       return count_fig2, mileage_fig, sp_fig







if __name__ == '__main__':
    app.run_server(port=8080, debug=True)