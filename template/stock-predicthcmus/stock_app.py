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


FA = "https://use.fontawesome.com/releases/v5.8.1/css/all.css"
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP, FA])
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
    "background-color": "#f8f9fa",
}

CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

MAIN_STYLE = {
    "margin-left": "18rem"
}

radioitemsPP = dbc.FormGroup(
    [
        dbc.RadioItems(
            options=[
                {"label": "XGBoost", "value": "XGBoost"},
                {"label": "RNN", "value": "RNN"},
                {"label": "LSTM", "value": "LSTM"},
                {"label": "ARIMA", "value": "ARIMA"},
                {"label": "Transformer and Time Embeddings", "value": "TSF & TE"},
            ],
            value="XGBoost",
            id="radioitems-input",
        ),
    ]
)

submenu_1 = [
    html.Li(
        # use Row and Col components to position the chevrons
        dbc.Row(
            [
                dbc.Col("Phương pháp", style={"fontWeight": "bold"}),
                dbc.Col(
                    html.I(className="fas fa-chevron-right mr-3"), width="auto"
                ),
            ],
            className="my-1",
        ),
        style={"cursor": "pointer"},
        id="submenu-1",
    ),
    # we use the Collapse component to hide and reveal the navigation links
    dbc.Collapse(
        [
            dbc.Form([radioitemsPP]),
        ],
        id="submenu-1-collapse",
    ),
]

checklistDT = dbc.FormGroup(
    [
        dbc.Checklist(
            options=[
                {"label": "Close", "value": 1},
                {"label": "Price Of Change", "value": 2}
            ],
            value=[1],
            id="checklist-input",
        ),
    ]
)


submenu_2 = [
    html.Li(
        dbc.Row(
            [
                dbc.Col("Đặc trưng", style={"fontWeight": "bold"}),
                dbc.Col(
                    html.I(className="fas fa-chevron-right mr-3"), width="auto"
                ),
            ],
            className="my-1",
        ),
        style={"cursor": "pointer"},
        id="submenu-2",
    ),
    dbc.Collapse(
        [
            dbc.Form([checklistDT]),
        ],
        id="submenu-2-collapse",
    ),
]

app.layout = html.Div([
    # side bar
    html.Div([
        html.H2("Predict Dashboard",  style={"textAlign": "center"}),
        html.Hr(),
        dbc.Nav(submenu_1 + submenu_2, vertical=True),
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
                         options=[{'label': 'Apple', 'value': 'AAPL'},
                                  {'label': 'Tesla', 'value': 'TSLA'},
                                  {'label': 'IBM', 'value': 'IBM'}],
                         multi=True, value=['AAPL'],
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


for i in [1, 2]:
    app.callback(
        Output(f"submenu-{i}-collapse", "is_open"),
        [Input(f"submenu-{i}", "n_clicks")],
        [State(f"submenu-{i}-collapse", "is_open")],
    )(toggle_collapse)

    app.callback(
        Output(f"submenu-{i}", "className"),
        [Input(f"submenu-{i}-collapse", "is_open")],
    )(set_navitem_class)


@app.callback(
    Output("dash-title", "children"),
    [
        Input("radioitems-input", "value"),
        Input("checklist-input", "value"),
    ],
)
def on_form_change(radio_items_value, checklist_value):
    template = 'Phương pháp {} - Đặc trưng [{}]'
    temp = ['Close', 'Price Of Change']
    title = ''
    for i in range(len(checklist_value)):
        if(i == 0):
            title += temp[checklist_value[i]-1]
        else:
            title += ', ' + temp[checklist_value[i]-1]

    # load file & update graph
    # TODO

    output_string = template.format(
        radio_items_value,
        title,
    )
    return output_string


@app.callback(Output('compare', 'figure'),
              [
    Input('my-dropdown', 'value'),
    Input("radioitems-input", "value"),
    Input("checklist-input", "value"),
])
def update_graph(selected_dropdown, radio_items_value, checklist_value):
    dropdown = {"TSLA": "Tesla", "AAPL": "Apple", "IBM": "IBM", }
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        # TODO load file
        method = ''
        for i in checklist_value:
            method += str(i)
        filename = './out/'+stock + '_' + radio_items_value + '_' + method + '.csv'

        # df = pd.read_csv("./out/AAPL_ARIMA_1.csv")
        df = pd.read_csv(filename)
        df.head()
        df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
        df.index = df['Date']

        # Load figure
        trace1.append(
            go.Scatter(x=df.index,
                       y=df["Prediction"],
                       mode='lines', opacity=0.7,
                       name=f'Prediction {dropdown[stock]}', textposition='bottom center'))
        trace2.append(
            go.Scatter(x=df.index,
                       y=df["Close"],
                       mode='lines', opacity=0.6,
                       name=f'Actual {dropdown[stock]}', textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1',
                                            '#FF7400', '#FFF400', '#FF0056'],
                                  height=600,
                                  title=f"Actual and Predict Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
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
