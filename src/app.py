import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import pandas as pd
import joblib
import numpy as np

# Load trained model
model = joblib.load("../artifact/model_1w.pkl") 

# Load raw dataset
df = pd.read_csv("../Data/005930.KS_weekly.csv")

app = dash.Dash(__name__)
server = app.server

# Layout
app.layout = html.Div([
    html.H1("Samsung Stock Direction Predictor", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Weeks Ahead to Predict:"),
        dcc.Slider(
            id='weeks-slider',
            min=1,
            max=8,
            step=1,
            value=1,
            marks={i: f"{i}w" for i in range(1, 9)},
        ),
    ], style={'margin': '20px'}),

    html.Button('Predict Latest', id='predict-btn', n_clicks=0, style={'display': 'block', 'margin': '20px auto'}),

    html.Div(id='prediction-output', style={'fontWeight': 'bold', 'textAlign': 'center'})
])


@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('weeks-slider', 'value')
)
def predict(n_clicks, weeks_ahead):
    if n_clicks == 0:
        return ""

    # Adjust features dynamically based on weeks_ahead
    window_short = 3 * weeks_ahead
    window_long = 5 * weeks_ahead

    df_sorted = df.sort_values('Date')  # sorts dataset
    df_latest = df_sorted.iloc[-1]

    # Compute moving averages, stds, and returns from the dataset
    features = pd.DataFrame([{
        'Close_ma3': df_sorted['Close'].rolling(window_short).mean().iloc[-1],
        'Close_ma5': df_sorted['Close'].rolling(window_long).mean().iloc[-1],
        'Volume_ma3': df_sorted['Volume'].rolling(window_short).mean().iloc[-1],
        'Volume_ma5': df_sorted['Volume'].rolling(window_long).mean().iloc[-1],
        'Close_std3': df_sorted['Close'].rolling(window_short).std().iloc[-1],
        'Close_std5': df_sorted['Close'].rolling(window_long).std().iloc[-1],
        'Return_1w': df_sorted['Close'].pct_change(5 * 1).iloc[-1],
        'Return_3w': df_sorted['Close'].pct_change(5 * 3).iloc[-1],
        'Close': df_latest['Close'],
        'Open': df_latest['Open'],
        'High': df_latest['High'],
        'Low': df_latest['Low']
    }])

    # Predict
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    result = "⬆️ Price will go UP" if prediction == 1 else "⬇️ Price will go DOWN"
    return f"Prediction ({weeks_ahead} week(s) ahead): {result} (Confidence: {probability:.2%})"

if __name__ == '__main__':
    app.run_server(debug=True)