
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import pandas as pd
import joblib
import numpy as np

# Load trained model
model = joblib.load("../artifact/model_1w.pkl") 

# Feature names used in training
feature_names = [
    'Close_ma3', 'Close_ma5',
    'Volume_ma3', 'Volume_ma5',
    'Close_std3', 'Close_std5',
    'Return_1w', 'Return_3w',
    'Close', 'Open', 'High', 'Low'
]

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Samsung Stock Direction Predictor (1 Week Ahead)", style={'textAlign': 'center'}),

    html.Div([
        html.Div([
            html.Label(f"{feature}"),
            dcc.Input(id=feature, type='number', step=0.01, value=0.0)
        ], style={'marginBottom': '10px'}) for feature in feature_names
    ]),

    html.Div([
        html.Button('Predict', id='predict-btn', n_clicks=0)
    ], style={'textAlign': 'center', 'margin': '20px'}),

    html.Div(id='prediction-output', style={'fontWeight': 'bold', 'textAlign': 'center'})
], style={'width': '60%', 'margin': 'auto', 'padding': '20px'})

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    [State(feature, 'value') for feature in feature_names]
)
def predict_direction(n_clicks, *feature_values):
    if n_clicks == 0:
        return ""

    input_data = pd.DataFrame([feature_values], columns=feature_names)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    result = "⬆️ Price will go UP next week" if prediction == 1 else "⬇️ Price will go DOWN next week"
    return f"Prediction: {result} (Confidence: {probability:.2%})"

if __name__ == '__main__':
    app.run_server(debug=True)
