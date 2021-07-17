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

# Read data from CSV file
df_nse = pd.read_csv("./out/stock_data_example.csv")

df_nse["Date"] = pd.to_datetime(df_nse.Date, format="%Y-%m-%d")
df_nse.index = df_nse['Date']


data = df_nse.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0, len(df_nse)), columns=['Date', 'Close'])

for i in range(0, len(data)):
    new_data["Date"][i] = data['Date'][i]
    new_data["Close"][i] = data["Close"][i]

new_data.index = new_data.Date
new_data.drop("Date", axis=1, inplace=True)

dataset = new_data.values

train = dataset[0:987, :]
valid = dataset[987:, :]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []

for i in range(60, len(train)):
    x_train.append(scaled_data[i-60:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = load_model("saved_model.h5")

inputs = new_data[len(new_data)-len(valid)-60:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

train = new_data[:987]
valid = new_data[987:]
valid['Predictions'] = closing_price


df = pd.read_csv("./out/stock_data_example.csv")

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
                {"label": "Transformer and Time Embeddings", "value": "TSF & TE"},
            ],
            value=1,
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
        html.H1(id="dash-title",
                style={"textAlign": "center"}),
        html.Div([
            html.H2("Actual closing price", style={"textAlign": "center"}),
            dcc.Graph(
                id="Actual Data",
                figure={
                    "data": [
                        go.Scatter(
                            x=train.index,
                            y=valid["Close"],
                            mode='markers'
                        )
                    ],
                    "layout":go.Layout(
                        title='scatter plot',
                        xaxis={'title': 'Date'},
                        yaxis={'title': 'Closing Rate'}
                    )
                }
            ),
            html.H2("LSTM Predicted closing price",
                    style={"textAlign": "center"}),
            dcc.Graph(
                id="Predicted Data",
                figure={
                    "data": [
                        go.Scatter(
                            x=valid.index,
                            y=valid["Predictions"],
                            mode='markers'
                        )

                    ],
                    "layout":go.Layout(
                        title='scatter plot',
                        xaxis={'title': 'Date'},
                        yaxis={'title': 'Closing Rate'}
                    )
                }

            )
        ]),
    ],
        style=MAIN_STYLE,
    ),
])


@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"TSLA": "Tesla", "AAPL": "Apple",
                "FB": "Facebook", "MSFT": "Microsoft", }
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(
            go.Scatter(x=df[df["Stock"] == stock]["Date"],
                       y=df[df["Stock"] == stock]["High"],
                       mode='lines', opacity=0.7,
                       name=f'High {dropdown[stock]}', textposition='bottom center'))
        trace2.append(
            go.Scatter(x=df[df["Stock"] == stock]["Date"],
                       y=df[df["Stock"] == stock]["Low"],
                       mode='lines', opacity=0.6,
                       name=f'Low {dropdown[stock]}', textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1',
                                            '#FF7400', '#FFF400', '#FF0056'],
                                  height=600,
                                  title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
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


@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value):
    dropdown = {"TSLA": "Tesla", "AAPL": "Apple",
                "FB": "Facebook", "MSFT": "Microsoft", }
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
            go.Scatter(x=df[df["Stock"] == stock]["Date"],
                       y=df[df["Stock"] == stock]["Volume"],
                       mode='lines', opacity=0.7,
                       name=f'Volume {dropdown[stock]}', textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1',
                                            '#FF7400', '#FFF400', '#FF0056'],
                                  height=600,
                                  title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
                                  xaxis={"title": "Date",
                                         'rangeselector': {'buttons': list([{'count': 1, 'label': '1M',
                                                                             'step': 'month',
                                                                             'stepmode': 'backward'},
                                                                            {'count': 6, 'label': '6M',
                                                                             'step': 'month',
                                                                             'stepmode': 'backward'},
                                                                            {'step': 'all'}])},
                                         'rangeslider': {'visible': True}, 'type': 'date'},
                                  yaxis={"title": "Transactions Volume"})}
    return figure

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
    template = "Phương pháp {} - {} đặc trưng"

    n_checkboxes = len(checklist_value)

    output_string = template.format(
        radio_items_value,
        n_checkboxes,
    )
    return output_string


if __name__ == '__main__':
    app.run_server(debug=True)
