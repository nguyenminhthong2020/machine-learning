import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np


Font = "https://use.fontawesome.com/releases/v5.8.1/css/all.css"
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP, Font])
server = app.server

scaler = MinMaxScaler(feature_range=(0, 1))

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#7fa934",
}

CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

MAIN_STYLE = {
    "margin-left": "18rem"
}

method = [
    html.Li(
        # use Row and Col components to position the chevrons
        dbc.Row(
            [
                dbc.Col("Method", style={"fontWeight": "bold"}),
                dbc.Col(
                    html.I(className="fas fa-chevron-right mr-3"), width="auto"
                ),
            ],
            className="my-1",
        ),
        style={"cursor": "pointer"},
        id="method",
    ),
    # we use the Collapse component to hide and reveal the navigation links
    dbc.Collapse(
        [
            dbc.Form([
                dbc.FormGroup(
                    [
                        dbc.RadioItems(
                            options=[
                                {"label": "LSTM", "value": "LSTM"},
                                {"label": "RNN", "value": "RNN"},
                                {"label": "XGBoost", "value": "XGBoost"},
                            ],
                            value="LSTM",
                            id="radio-items",
                        ),
                    ]
                )
            ]),
        ],
        id="collapse_method",
    ),
]

feature = [
    html.Li(
        dbc.Row(
            [
                dbc.Col("Feature", style={"fontWeight": "bold"}),
                dbc.Col(
                    html.I(className="fas fa-chevron-right mr-3"), width="auto"
                ),
            ],
            className="my-1",
        ),
        style={"cursor": "pointer"},
        id="feature",
    ),
    dbc.Collapse(
        [
            dbc.Form([
                dbc.FormGroup([
                    dbc.Checklist(
                        options=[
                            {"label": "Price Of Close", "value": 1},
                            {"label": "Price Of Change", "value": 2}
                        ],
                        value=[1],
                        id="checklist-items",
                    ),
                ]
            )]),
        ],
        id="collapse_feature",
    ),
]

app.layout = html.Div([
    # side bar
    html.Div([
        html.H2("MENU",  style={"textAlign": "center"}),
        html.Hr(),
        dbc.Nav(method + feature, vertical=True),
    ],
        style=SIDEBAR_STYLE,
        id="sidebar",
    ),


    html.Div(id="page-content", style=CONTENT_STYLE),

    html.Div([
        html.H1("Stock Price Analysis Dashboard",
                style={"textAlign": "center"}),
        html.Hr(),
        html.Div([
            html.H3(id="dash-title",
                    style={"textAlign": "center"}),

            dcc.Dropdown(id='my-dropdown',
                         options=[{'label': '^IXIC', 'value': '^IXIC'},
                                  {'label': 'NIO', 'value': 'NIO'},
                                  {'label': 'OGEN', 'value': 'OGEN'},
                                  {'label': 'UPS', 'value': 'UPS'},
                                  {'label': 'XPEV', 'value': 'XPEV'},],
                         multi=True, value=['^IXIC'],
                         style={"display": "block", "margin-left": "auto",
                                "margin-right": "auto", "width": "60%"}),
             dcc.Graph(id='compare'),
             ], className="container"),
    ],
        style=MAIN_STYLE,
    ),
])

# this function is used to toggle the is_open property of each Collapse
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

# this function applies the "open" class to rotate the chevron
def set_navitem_class(is_open):
    if is_open:
        return "open"
    return ""

# callback method
app.callback(
    Output(f"collapse_method", "is_open"),
    [Input(f"method", "n_clicks")],
    [State(f"collapse_method", "is_open")],
)(toggle_collapse)

app.callback(
    Output(f"method", "className"),
    [Input(f"collapse_method", "is_open")],
)(set_navitem_class)

# callback feature
app.callback(
    Output(f"collapse_feature", "is_open"),
    [Input(f"feature", "n_clicks")],
    [State(f"collapse_feature", "is_open")],
)(toggle_collapse)

app.callback(
    Output(f"feature", "className"),
    [Input(f"collapse_feature", "is_open")],
)(set_navitem_class)

@app.callback(Output('compare', 'figure'), [
    Input('my-dropdown', 'value'),
    Input("radio-items", "value"),
    Input("checklist-items", "value"),
])
def update_graph(selected_dropdown, radio_items_value, checklist_value):
    dropdown = { "^IXIC": "^IXIC", "NIO": "NIO", "OGEN": "OGEN", "UPS": "UPS", "XPEV": "XPEV"}
    trace_predict = []
    trace_original = []
    for stock in selected_dropdown:
        # TODO load file
        filename = './out/' + radio_items_value + '/' + stock + '.csv'

        df = pd.read_csv(filename)
        df.head()
        df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
        df.index = df['Date']

        # Load figure
        trace_predict.append(
            go.Scatter(x=df.index,
                       y=df["Prediction"],
                       mode='lines',
                       name=f'Predicted price of {dropdown[stock]}', textposition='bottom center'))
        trace_original.append(
            go.Scatter(x=df.index,
                       y=df["Close"],
                       mode='lines',
                       name=f'Original price of {dropdown[stock]}', textposition='bottom center'))
    traces = [trace_original, trace_predict]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#f61111", '#00ff51', '#f0ff00',
                                            '#8900ff', '#00d2ff', '#ff7400'],
                                  height=600,
                                  xaxis={"title": "Date",
                                         'rangeselector': {'buttons': list([{'count': 1, 'label': '1M',
                                                                             'step': 'month',
                                                                             'stepmode': 'backward'},
                                                                            {'count': 6, 'label': '6M',
                                                                             'step': 'month',
                                                                             'stepmode': 'backward'},
                                                                            {'step': 'all'}])},
                                         'rangeslider': {'visible': True}, 'type': 'date'},
                                  yaxis={"title": "Price (USD)"})}
    return figure


if __name__ == '__main__':
    app.run_server(debug=True)
