import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

df = pd.read_csv('Model_Comparison.csv')

df = df.sample(250)
df = df.sort_values('Actual', ascending = True).reset_index(drop=True)




fig = go.Figure()

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

controls = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label("Model Options"),
                dcc.Dropdown(
                    id="model_select",
                    options=[
                        {'label': 'Large Data set', 'value': 1},
                        {'label': 'Small Data Set', 'value': 2},

                    ],value=1,

                ),
            ]
        ),

    ],
    body=True,
)

app.layout = dbc.Container(
    [
        html.H1("Models vs actual"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(controls, md=3),
                dbc.Col(dcc.Graph(id="model_graph"), md=8),
            ],

            align="center",
        ),
    ],
    fluid=True,
)
@app.callback(
    Output("model_graph", "figure"),
    [Input("model_select", "value"),
    ],)

def test (model_select):
    t='markers'
    if model_select == 1:
        data =[

            go.Scatter(
                y=df.iloc[:,1],
                mode=t,
                name="RF_80% Acurate model",
            ),
            go.Scatter(
                y=df.iloc[:, 3],
                mode=t,
                name="OLS_80% Acurate mode",
            ),
            go.Scatter(
                y=df.iloc[:, 0],
                mode="lines",
                marker={"size": 8},
                name="Actual(base)"
            ),
        ]
    else:
        data = [

            go.Scatter(
                y=df.iloc[:, 2],
                mode=t,
                name="RF_60% Acurate model",
            ),
            go.Scatter(
                y=df.iloc[:, 4],
                mode=t,
                name="OLS_60% Acurate model",
            ),
            go.Scatter(
                y=df.iloc[:, 0],
                mode="lines",
                marker={"size": 8},
                name="Actual(base)"
            ),
        ]

    layout = {"yaxis": {"title": "Plot of Models"}}
    return  go.Figure(data=data, layout=layout)

if __name__ == "__main__":
    app.run_server(debug=True, port=8888) # Turn off reloader if inside Jupyter
