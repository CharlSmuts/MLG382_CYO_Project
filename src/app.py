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
Weekly_df = pd.read_csv("../Data/005930.KS_weekly.csv")
Monthly_df = pd.read_csv("../Data/005930.KS_monthly.csv")

# Convert 'Date' columns to datetime
Daily_df['Date'] = pd.to_datetime(Daily_df['Date'])
Weekly_df['Date'] = pd.to_datetime(Weekly_df['Date'])
Monthly_df['Date'] = pd.to_datetime(Monthly_df['Date'])

# Sort by date just to be safe
Daily_df.sort_values("Date", inplace=True)
Weekly_df.sort_values("Date", inplace=True)
Monthly_df.sort_values("Date", inplace=True)

# Declare app
app = dash.Dash(__name__)
server = app.server

# Layout
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

# Define Inputs and outputs
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
    "Daily": {
        "df": Daily_df,
        "model": "../Artifact/Daily_stock_price_prediction_model_2.h5",
        "scaler_X": "../Artifact/scaler_X_daily.pkl",
        "scaler_y": "../Artifact/scaler_y_daily.pkl",
        "lookback": 30,
        "date_increment": datetime.timedelta(days=1)
    },
    "Weekly": {
        "df": Weekly_df,
        "model": "../Artifact/Weekly_stock_price_prediction_model_2.h5",
        "scaler_X": "../Artifact/scaler_X_weekly.pkl",
        "scaler_y": "../Artifact/scaler_y_weekly.pkl",
        "lookback": 12,
        "date_increment": datetime.timedelta(weeks=1)
    },
    "Monthly": {
        "df": Monthly_df,
        "model": "../Artifact/Monthly_stock_price_prediction_model_2.h5",
        "scaler_X": "../Artifact/scaler_X_monthly.pkl",
        "scaler_y": "../Artifact/scaler_y_monthly.pkl",
        "lookback": 6,
        "date_increment": datetime.timedelta(weeks=4)  # approx 1 month
    }
}

            if scale not in data_map:
                return f"No model/data available for scale '{scale}'."

            selected = data_map[scale]
            df = selected["df"]
            model = load_model(selected["model"])
            scaler_X = joblib.load(selected["scaler_X"])
            scaler_y = joblib.load(selected["scaler_y"])
            lookback = selected["lookback"]
            date_increment = selected["date_increment"]

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

            # Loop to predict until reaching the target date
            current_date = closest_row['Date']
            future_date = input_date + date_increment * period
            predicted_price = adj_close_at_date
            total_shares = shares

            while current_date < future_date:
                # Predict the next price
                scaled_prediction = model.predict(latest_input)
                predicted_adj_close = scaler_y.inverse_transform(scaled_prediction)[0][0]

                # Update the current date
                current_date += date_increment
                predicted_price = predicted_adj_close

                # Prepare input for the next prediction
                input_seq = np.roll(input_seq, -1, axis=0)
                input_seq[-1] = np.array([df.loc[df['Date'] == current_date, feature_cols].values[0]])
                input_seq_scaled = scaler_X.transform(input_seq)

                latest_input = input_seq_scaled[-1].reshape(1, -1)

            # Final investment value
            future_value = predicted_price * total_shares

            return html.Div([
                f"On {closest_row['Date'].date()}, you could buy {shares:.2f} shares at "
                f"${adj_close_at_date:.2f} each.",
                html.Br(),
                f"Predicted Adj Close after {period} {scale.lower()}(s): ${predicted_price:.2f}",
                html.Br(),
                f"Your investment could be worth: ${future_value:,.2f}"
            ])

        except Exception as e:
            return f"Prediction error: {str(e)}"

    return ""
