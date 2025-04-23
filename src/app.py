# Import Modules
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
import datetime
import joblib

# Load Raw data sets
Daily_df = pd.read_csv("../Data/005930.KS.csv")
Monthly_df = pd.read_csv("../Data/005930.KS_monthly.csv")

# Convert 'Date' columns to datetime
Daily_df['Date'] = pd.to_datetime(Daily_df['Date'])
Monthly_df['Date'] = pd.to_datetime(Monthly_df['Date'])

# Sort by date just to be safe
Daily_df.sort_values("Date", inplace=True)
Monthly_df.sort_values("Date", inplace=True)

# Declare app
app = dash.Dash(__name__)
server = app.server

# Layout
app.layout = html.Div([
    html.H1("Welcome to the Samsung Stock Price Predictor", style={'color': '#0077cc'}),

    html.Label("Select Graph:"),
    dcc.Dropdown(
        id='graphSelect',
        options=[
            {'label': 'Daily', 'value': 'Daily'},
            {'label': 'Monthly', 'value': 'Monthly'}
        ],
        value='Daily'
    ),

    html.Div([
        html.Img(id='Daily', src='assets/DailyGraph.png', style={'display': 'none', 'width': '100%', 'marginTop': '20px'}),
        html.Img(id='Monthly', src='assets/MonthlyGraph.png', style={'display': 'none', 'width': '100%', 'marginTop': '20px'}),
    ], id='graphs'),

    html.Div([
        html.Label("Initial investment amount:"),
        dcc.Input(id='InitialInvestment', type='number', placeholder='Enter amount', style={'width': '96.5%', 'padding': '8px'}),

        html.Label("Select time scale:"),
        dcc.Dropdown(
            id='timeScale',
            options=[
                {'label': 'Daily', 'value': 'Daily'},
                {'label': 'Monthly', 'value': 'Monthly'}
            ],
            value='Daily'
        ),

        html.Label("Please select a date for the initial investment:"),
        html.Br(),
        dcc.DatePickerSingle(
            id='InitialInvestmentDate',
            placeholder='Select a date',
            display_format='YYYY-MM-DD'
        ),
        html.Br(),

        html.Label("Investment period (in days/months, Max 30 days/5 months):"),
        dcc.Input(id='InvestmentPeriod', type='number', placeholder='Enter period', style={'width': '96.5%', 'padding': '8px'}),

        html.Button('Calculate Prediction', id='SubmitInvestment', n_clicks=0,
                    style={'marginTop': '10px', 'width': '100%', 'padding': '8px'}),
    ], id='PredictionBox', style={'marginTop': '30px'}),

    html.Div(id='prediction-output', style={'fontWeight': 'bold', 'marginTop': '20px'})
], style={
    'fontFamily': 'Arial, sans-serif',
    'padding': '20px',
    'maxWidth': '600px',
    'margin': 'auto'
})


# Update DatePicker max date based on timeScale
@app.callback(
    Output('InitialInvestmentDate', 'max_date_allowed'),
    Input('timeScale', 'value')
)
def update_max_date(scale):
    if scale == 'Daily':
        return Daily_df['Date'].max().date()
    elif scale == 'Monthly':
        return Monthly_df['Date'].max().date()
    return datetime.date.today()


# Prediction callback
@app.callback(
    Output('prediction-output', 'children'),
    Input('SubmitInvestment', 'n_clicks'),
    State('InitialInvestment', 'value'),
    State('timeScale', 'value'),
    State('InitialInvestmentDate', 'date'),
    State('InvestmentPeriod', 'value')
)
def update_prediction(n_clicks, investment, scale, date, period):
    if n_clicks > 0:
        try:
            investment = float(investment)
            period = int(period)

            if scale == "Daily" and period > 30:
                period = 30
            elif scale == "Monthly" and period > 5:
                period = 5


            if not date:
                return "Please select a valid start date."

            input_date = datetime.datetime.strptime(date, "%Y-%m-%d")

            unit_map = {"Daily": "days", "Weekly": "weeks", "Monthly": "months"}
            if scale not in unit_map:
                return f"No model/data available for scale '{scale}'."

            unit = unit_map[scale]

            data_map = {
                "Daily": {
                    "df": Daily_df,
                    "model": "../Artifact/Daily_stock_price_prediction_model_2.h5",
                    "scaler_X": "../Artifact/scaler_X_daily.pkl",
                    "scaler_y": "../Artifact/scaler_y_daily.pkl",
                    "lookback": 30,
                    "delta": datetime.timedelta(days=1)
                },
                "Monthly": {
                    "df": Monthly_df,
                    "model": "../Artifact/Monthly_stock_price_prediction_model_2.h5",
                    "scaler_X": "../Artifact/scaler_X_monthly.pkl",
                    "scaler_y": "../Artifact/scaler_y_monthly.pkl",
                    "lookback": 6,
                    "delta": pd.DateOffset(months=1)
                }
            }

            selected = data_map[scale]
            df = selected["df"]
            model = load_model(selected["model"])
            scaler_X = joblib.load(selected["scaler_X"])
            scaler_y = joblib.load(selected["scaler_y"])
            lookback = selected["lookback"]
            delta = selected["delta"]

            if input_date > df['Date'].max():
                return f"Please choose a date on or before {df['Date'].max().date()}."

            if input_date not in df['Date'].values:
                closest_idx = df['Date'].sub(input_date).abs().idxmin()
            else:
                closest_idx = df[df['Date'] == input_date].index[0]

            closest_row = df.loc[closest_idx]
            adj_close_at_date = closest_row['Adj Close']
            shares = investment / adj_close_at_date

            feature_cols = ['Open', 'High', 'Low', 'Volume']
            input_seq = df[feature_cols].iloc[closest_idx - lookback:closest_idx].values

            if input_seq.shape[0] < lookback:
                padding = np.zeros((lookback - input_seq.shape[0], len(feature_cols)))
                input_seq = np.vstack((padding, input_seq))

            input_seq_scaled = scaler_X.transform(input_seq)

            current_date = df.loc[closest_idx, 'Date']
            for _ in range(period):
                latest_input = input_seq_scaled.reshape(lookback, len(feature_cols))
                scaled_pred = model.predict(latest_input)
                pred_price = scaler_y.inverse_transform(scaled_pred)[0][0]

                last_known_features = input_seq[-1].copy()
                last_known_features[0:3] = pred_price
                input_seq_scaled = np.vstack((input_seq_scaled, last_known_features))
                input_seq_scaled = input_seq_scaled[-lookback:]

                current_date += delta if unit != "months" else pd.DateOffset(months=1)

            final_value = pred_price * shares

            return html.Div([
                f"On {closest_row['Date'].date()}, you could buy {shares:.2f} shares at "
                f"${adj_close_at_date:.2f} each.",
                html.Br(),
                f"Predicted Adj Close on {current_date.date()}: ${pred_price:.2f}",
                html.Br(),
                f"Your investment could be worth: ${final_value:,.2f}"
            ])

        except Exception as e:
            return f"Prediction error: {str(e)}"

    return ""


# Graph toggle callback
@app.callback(
    [Output('Daily', 'style'),
     Output('Monthly', 'style')],
    Input('graphSelect', 'value')
)
def toggle_graph_visibility(selected_graph):
    if selected_graph == 'Daily':
        return {'display': 'block', 'width': '100%', 'marginTop': '20px'}, {'display': 'none'}
    elif selected_graph == 'Monthly':
        return {'display': 'none'}, {'display': 'block', 'width': '100%', 'marginTop': '20px'}
    return {'display': 'none'}, {'display': 'none'}
