#impoort Modules
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
import datetime
import joblib

#load Raw data sets
Daily_df = pd.read_csv("../Data/005930.KS.csv")
Weekly_df = pd.read_csv("../Data/005930.KS_weekly.csv")
Monthly_df = pd.read_csv("../Data/005930.KS_monthly.csv")

scaler_X = joblib.load("../Artifact/scaler_X.pkl")
scaler_y = joblib.load("../Artifact/scaler_y.pkl")

# Convert 'Date' columns to datetime
Daily_df['Date'] = pd.to_datetime(Daily_df['Date'])
Weekly_df['Date'] = pd.to_datetime(Weekly_df['Date'])
Monthly_df['Date'] = pd.to_datetime(Monthly_df['Date'])

# Sort by date just to be safe
Daily_df.sort_values("Date", inplace=True)
Weekly_df.sort_values("Date", inplace=True)
Monthly_df.sort_values("Date", inplace=True)


#Declare app
app = dash.Dash(__name__)
server = app.server

#Layout
app.layout = html.Div([
    html.H1("Welcome to Samsung Stock Price Predictor", style={'color': '#0077cc'}),

    html.Label("Select Graph:"),
    dcc.Dropdown(
        id='graphSelect',
        options=[
            {'label': 'Daily', 'value': 'Daily'},
            {'label': 'Weekly', 'value': 'Weekly'},
            {'label': 'Monthly', 'value': 'Monthly'}
        ],
        value='Daily'
    ),

    html.Div([
        html.Img(id='Daily', src='/assets/DailyGraph.png', style={'display': 'none', 'width': '100%', 'marginTop': '20px'}),
        html.Img(id='Weekly', src='/assets/WeeklyGraph.png', style={'display': 'none', 'width': '100%', 'marginTop': '20px'}),
        html.Img(id='Monthly', src='/assets/MonthlyGraph.png', style={'display': 'none', 'width': '100%', 'marginTop': '20px'}),
    ], id='graphs'),

    html.Div([
        html.Label("Initial investment amount:"),
        dcc.Input(id='InitialInvestment', type='number', placeholder='Enter amount', style={'width': '96.5%', 'padding': '8px'}),

        html.Label("Select time scale:"),
        dcc.Dropdown(
            id='timeScale',
            options=[
                {'label': 'Daily', 'value': 'Daily'},
                {'label': 'Weekly', 'value': 'Weekly'},
                {'label': 'Monthly', 'value': 'Monthly'}
            ],
            value='Daily'
        ),

        html.Label("Please select a date for the initial investment:"),
        dcc.DatePickerSingle(
            id='InitialInvestmentDate',
            placeholder='Select a date',
            display_format='YYYY-MM-DD'
        ),

        html.Label("Investment period (in days/weeks/months):"),
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

#Define Inputs and outputs
@app.callback(
    Output('prediction-output', 'children'),
    Input('SubmitInvestment', 'n_clicks'),
    State('InitialInvestment', 'value'), # value
    State('timeScale', 'value'),
    State('InitialInvestmentDate', 'date'),
    State('InvestmentPeriod', 'value') # value
)
def update_prediction(n_clicks, investment, scale, date, period):
    if n_clicks > 0:
        try:
            investment = float(investment)
            period = int(period)
            if not date:
                return "Please select a valid start date."
            
            input_date = datetime.datetime.strptime(date, "%Y-%m-%d")

            # Select dataset and model based on time scale
            data_map = {
                "Daily": (Daily_df, "../Artifact/Daily_stock_price_prediction_model_2.h5", 30),
                "Weekly": (Weekly_df, "../Artifact/Weekly_stock_price_prediction_model_2.h5", 12),
                "Monthly": (Monthly_df, "../Artifact/Monthly_stock_price_prediction_model_2.h5", 6)
            }

            if scale not in data_map:
                return f"No model/data available for scale '{scale}'."

            df, model_path, lookback = data_map[scale]
            model = load_model(model_path)

            # Find the closest available date
            closest_idx = df['Date'].sub(input_date).abs().idxmin()
            closest_row = df.loc[closest_idx]
            adj_close_at_date = closest_row['Adj Close']

            # Calculate number of shares bought
            shares = investment / adj_close_at_date

            # Prepare feature input sequence
            feature_cols = ['Open', 'High', 'Low', 'Volume']
            start_idx = max(0, closest_idx - lookback)
            input_seq = df[feature_cols].iloc[start_idx:closest_idx].values

            # Pad sequence if too short
            if input_seq.shape[0] < lookback:
                padding = np.zeros((lookback - input_seq.shape[0], len(feature_cols)))
                input_seq = np.vstack((padding, input_seq))

            # Scale input using the same scaler used during training
            input_seq_scaled = scaler_X.transform(input_seq)

            # Make sure input shape is correct: (lookback, num_features)
            assert input_seq_scaled.shape == (lookback, len(feature_cols)), \
                f"Expected shape ({lookback}, {len(feature_cols)}), got {input_seq_scaled.shape}"

            latest_input = input_seq_scaled[-1].reshape(1, -1)

            # Predict scaled output and inverse-transform
            scaled_prediction = model.predict(latest_input)
            predicted_adj_close = scaler_y.inverse_transform(scaled_prediction)[0][0]

            # Final investment value
            future_value = predicted_adj_close * shares

            return (
                f"On {closest_row['Date'].date()}, you could buy {shares:.2f} shares at "
                f"${adj_close_at_date:.2f} each.\n"
                f"Predicted Adj Close after {period} {scale.lower()}(s): ${predicted_adj_close:.2f}\n"
                f"Your investment could be worth: ${future_value:,.2f}"
            )

        #except ValueError:
        #    return "Please enter valid numeric values for investment and period."
        except Exception as e:
            return f"Prediction error: {str(e)}"

    return ""