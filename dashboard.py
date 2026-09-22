import dash
from dash import html, dcc, Input, Output, State, callback_context
import plotly.express as px
import pandas as pd
import requests
import os
from dash import dash_table
import json
import base64
from io import BytesIO
from predictor import LANGUAGE_NAMES # Import the language mapping

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5000")

app = dash.Dash(__name__)
app.title = "AI vs Human Real-Time Dashboard"

# Define common card style
card_style = {
    'border': '1px solid #e0e0e0',
    'borderRadius': '12px',
    'boxShadow': '0 4px 12px rgba(0,0,0,0.05)',
    'padding': '25px',
    'marginBottom': '30px',
    'backgroundColor': 'white',
    'transition': 'all 0.3s ease-in-out'
}

app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'padding': '20px', 'backgroundColor': '#f7f9fc'}, children=[
    html.H1("AI vs Human Text Detector", style={'textAlign': 'center', 'color': '#333', 'marginBottom': '40px', 'fontSize': '2.5em'}),

    # User Input & Prediction Section
    html.Div(style=card_style, children=[
        html.H3("Try Your Own Text", style={'color': '#444', 'borderBottom': '1px solid #eee', 'paddingBottom': '10px', 'marginBottom': '20px'}),
        dcc.Textarea(
            id='user-input',
            placeholder='Enter a sentence...',
            style={'width': '100%', 'height': 120, 'borderRadius': '8px', 'border': '1px solid #ddd', 'padding': '15px', 'fontSize': '1.1em', 'boxShadow': 'inset 0 1px 3px rgba(0,0,0,0.08)'}
        ),
        html.Div(id='char-count', style={'textAlign': 'right', 'fontSize': '0.9em', 'color': '#666', 'marginTop': '5px'}),
        
        # Input Examples
        html.Div([
            html.Span("Try some examples: ", style={'marginRight': '10px', 'fontWeight': 'bold', 'color': '#555'}),
            html.Button("AI Example", id='example-ai-button', n_clicks=0, style={'marginRight': '10px', 'padding': '8px 15px', 'border': '1px solid #007bff', 'borderRadius': '5px', 'backgroundColor': 'white', 'cursor': 'pointer'}),
            html.Button("Human Example", id='example-human-button', n_clicks=0, style={'marginRight': '10px', 'padding': '8px 15px', 'border': '1px solid #28a745', 'borderRadius': '5px', 'backgroundColor': 'white', 'cursor': 'pointer'}),
            html.Button("Hindi Example", id='example-hindi-button', n_clicks=0, style={'padding': '8px 15px', 'border': '1px solid #ffc107', 'borderRadius': '5px', 'backgroundColor': 'white', 'cursor': 'pointer'}),
        ], style={'marginBottom': '20px', 'marginTop': '15px'}),

        html.Div([
            html.B("Prediction Strategy: ", style={'marginRight': '15px'}),
            dcc.RadioItems(
                id='prediction-strategy-toggle',
                options=[
                    {'label': 'Ensemble (Default)', 'value': 'ensemble'},
                    {'label': 'BERT Only', 'value': 'bert'},
                    {'label': 'Random Forest Only', 'value': 'rf'}
                ],
                value='ensemble',
                inline=True,
                style={'display': 'inline-block', 'marginBottom': '20px'},
                labelStyle={'display': 'inline-block', 'marginRight': '20px'}
            ),
        ]),
        html.Div([
            html.Button(
                'Submit Text',
                id='submit-button',
                n_clicks=0,
                style={
                    'backgroundColor': '#007bff', 'color': 'white', 'border': 'none',
                    'padding': '12px 25px', 'borderRadius': '8px', 'fontSize': '1.1em',
                    'cursor': 'pointer', 'marginTop': '15px', 'transition': 'background-color 0.3s'
                }
            ),
            # Clear Input Button
            html.Button(
                'Clear',
                id='clear-input-button',
                n_clicks=0,
                style={
                    'backgroundColor': '#6c757d', 'color': 'white', 'border': 'none',
                    'padding': '12px 25px', 'borderRadius': '8px', 'fontSize': '1.1em',
                    'cursor': 'pointer', 'marginTop': '15px', 'marginLeft': '10px', 'transition': 'background-color 0.3s'
                }
            ),
        ]),
        
        # Loading Indicator
        dcc.Loading(
            id="loading-output",
            type="circle",
            children=html.Div(id='prediction-result', style={'marginTop': '25px', 'fontWeight': 'bold', 'fontSize': '1.2em', 'lineHeight': '1.6'})
        ),
        
        html.Div(id='shap-top-tokens', style={'marginTop': '10px', 'marginBottom': '15px', 'fontSize': '0.95em', 'color': '#555'}),
        html.Div(id='feedback-section', style={'marginTop': '20px', 'borderTop': '1px solid #eee', 'paddingTop': '20px'}),
        dcc.Store(id='last-prediction-data'),
        
        # Notifications
        dcc.Store(id='notification-trigger', data={'type': None, 'message': None, 'id': 0}),
        html.Div(id='notification-container', style={'position': 'fixed', 'top': '20px', 'right': '20px', 'zIndex': 9999}),
        
    ]),

    # Dashboard Insights Section
    html.Div(style=card_style, children=[
        html.H3("Real-Time Analytics", style={'color': '#444', 'borderBottom': '1px solid #eee', 'paddingBottom': '10px', 'marginBottom': '20px'}),
        html.Div([
            html.Div([
                html.H4("Prediction Count", style={'textAlign': 'center', 'color': '#555'}),
                dcc.Graph(id="prediction-count", config={'displayModeBar': False})
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '2%'}),
            html.Div([
                html.H4("Language Distribution", style={'textAlign': 'center', 'color': '#555'}),
                dcc.Graph(id="language-distribution", config={'displayModeBar': False})
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        ]),
        html.Div([
            html.Div([
                html.H4("BERT Confidence Score by Label", style={'textAlign': 'center', 'color': '#555'}),
                dcc.Graph(id="confidence-boxplot", config={'displayModeBar': False})
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '2%'}),
            html.Div([
                html.H4("SHAP Word-Level Explanations", id="shap-chart-title", style={'textAlign': 'center', 'color': '#555'}),
                dcc.Graph(id="shap-words", config={'displayModeBar': False})
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        ]),
        # NEW: Sentiment Distribution Chart
        html.Div([
            html.Div([
                html.H4("Sentiment Distribution", style={'textAlign': 'center', 'color': '#555'}),
                dcc.Graph(id="sentiment-distribution", config={'displayModeBar': False})
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '2%'}),
            html.Div([
                html.H4("User Feedback Summary", style={'textAlign': 'center', 'color': '#555'}),
                dcc.Graph(id="feedback-summary-chart", config={'displayModeBar': False})
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        ]),
        html.Div([
            html.Div([
                html.H4("Download Data", style={'textAlign': 'center', 'color': '#555'}),
                html.Button(
                    'Download Feedback Logs (JSON)',
                    id='download-feedback-button',
                    n_clicks=0,
                    style={
                        'backgroundColor': '#28a745', 'color': 'white', 'border': 'none',
                        'padding': '12px 25px', 'borderRadius': '8px', 'fontSize': '1em',
                        'cursor': 'pointer', 'marginTop': '15px', 'transition': 'background-color 0.3s'
                    }
                ),
                dcc.Download(id="download-feedback-json"),
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingTop': '70px', 'textAlign': 'center'}),
        ]),
    ]),

    # Prediction History Section
    html.Div(style=card_style, children=[
        html.H3("Prediction History", style={'color': '#444', 'borderBottom': '1px solid #eee', 'paddingBottom': '10px', 'marginBottom': '20px'}),
        html.Div(id="prediction-history-table-container", style={'maxHeight': '400px', 'overflowY': 'scroll', 'border': '1px solid #eee', 'borderRadius': '8px'}),
    ]),

    dcc.Interval(id="interval-component", interval=5000, n_intervals=0)
])

# --- Callback for Character Counter ---
@app.callback(
    Output('char-count', 'children'),
    Input('user-input', 'value')
)
def update_char_count(text):
    if text:
        return f"{len(text)} characters"
    return "0 characters"

# --- Callbacks for Input Examples ---
@app.callback(
    Output('user-input', 'value', allow_duplicate=True),
    [Input('example-ai-button', 'n_clicks'),
     Input('example-human-button', 'n_clicks'),
     Input('example-hindi-button', 'n_clicks')],
    prevent_initial_call=True
)
def set_example_text(ai_n, human_n, hindi_n):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'example-ai-button':
        return "The rapid advancement of artificial intelligence is revolutionizing industries globally, enhancing efficiency and enabling unprecedented innovation."
    elif button_id == 'example-human-button':
        return "Hey, what's up? Just chilling here, thinking about grabbing a coffee later. Wanna join?"
    elif button_id == 'example-hindi-button':
        return "आज का मौसम बहुत सुहावना है। मुझे लगता है कि यह घूमने के लिए एक अच्छा दिन है।"
    return dash.no_update

# --- Callback for Clear Input Button ---
@app.callback(
    Output('user-input', 'value', allow_duplicate=True),
    Input('clear-input-button', 'n_clicks'),
    prevent_initial_call=True
)
def clear_input_text(n_clicks):
    if n_clicks > 0:
        return ""
    return dash.no_update

#  Main Prediction Callback 
@app.callback(
    [Output('prediction-result', 'children'),
     Output('shap-top-tokens', 'children'),
     Output('feedback-section', 'children'),
     Output('last-prediction-data', 'data'),
     Output('notification-trigger', 'data', allow_duplicate=True)],
    [Input('submit-button', 'n_clicks')],
    [State('user-input', 'value'),
     State('prediction-strategy-toggle', 'value')],
    prevent_initial_call=True
)
def perform_prediction(n_clicks, user_input, strategy):
    if n_clicks > 0 and user_input:
        try:
            response = requests.post(f"{BACKEND_URL}/predict", json={"texts": [user_input], "strategy": strategy})
            response.raise_for_status()
            prediction_data = response.json()[0]

            final_label = prediction_data["Final Label"]
            bert_conf = prediction_data["BERT Conf"]
            rf_conf = prediction_data["RF Conf"]
            detected_language_code = prediction_data["Language"]
            translated_text = prediction_data.get("Translated Text", None)
            response_time = prediction_data["Response Time (ms)"]
            shap_words = prediction_data["SHAP Words"]
            original_text = prediction_data["Original Text"]
            # NEW: Get sentiment data
            sentiment_label = prediction_data.get("Sentiment Label", "N/A")
            sentiment_score = prediction_data.get("Sentiment Score", 0.0)

            # Get full language name
            detected_language_name = LANGUAGE_NAMES.get(detected_language_code, detected_language_code.upper())

            label_color = "#dc3545" if final_label == "AI" else "#28a745"
            
            # Prediction Result Display
            result_display_children = [
                html.Span("Prediction: ", style={'fontWeight': 'normal'}),
                html.Span(
                    final_label,
                    style={
                        'backgroundColor': label_color,
                        'color': 'white',
                        'padding': '8px 15px',
                        'borderRadius': '5px',
                        'marginLeft': '10px',
                        'display': 'inline-block',
                        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                    }
                ),
                html.Span(f" (Language: {detected_language_name})", style={'fontSize': '0.9em', 'color': '#666', 'marginLeft': '10px'}),
                html.Br(),
            ]

            # Add translated text if available and not English
            if translated_text and detected_language_code != 'en':
                result_display_children.append(
                    html.Div([
                        html.Span("Translated to English: ", style={'fontWeight': 'normal', 'marginTop': '10px', 'display': 'inline-block', 'fontStyle': 'italic'}),
                        html.Span(translated_text, style={'display': 'inline-block', 'marginLeft': '5px', 'fontStyle': 'italic', 'color': '#555'})
                    ], style={'marginTop': '10px'})
                )
            
            # NEW: Add Sentiment display
            sentiment_color_map = {'POSITIVE': '#28a745', 'NEGATIVE': '#dc3545', 'NEUTRAL': '#ffc107', 'N/A': '#6c757d', 'Error': '#6c757d'}
            result_display_children.append(
                html.Div([
                    html.Span("Sentiment: ", style={'fontWeight': 'normal', 'marginTop': '10px', 'display': 'inline-block'}),
                    html.Span(
                        f"{sentiment_label} ({sentiment_score:.2f}%)",
                        style={
                            'backgroundColor': sentiment_color_map.get(sentiment_label, '#6c757d'),
                            'color': 'white',
                            'padding': '5px 10px',
                            'borderRadius': '5px',
                            'marginLeft': '10px',
                            'display': 'inline-block',
                            'fontSize': '0.9em'
                        }
                    )
                ], style={'marginTop': '5px'})
            )
            
            # Append common confidence and response time elements
            result_display_children.extend([
                html.Div([
                    html.Span("BERT Confidence: ", style={'fontWeight': 'normal', 'marginTop': '10px', 'display': 'inline-block'}),
                    html.Div([
                        html.Div(f"{bert_conf}%", style={'width': f"{bert_conf}%", 'backgroundColor': '#007bff', 'height': '20px', 'lineHeight': '20px', 'color': 'white', 'textAlign': 'right', 'paddingRight': '5px', 'borderRadius': '3px'})
                    ], style={'width': '200px', 'backgroundColor': '#e9ecef', 'borderRadius': '3px', 'display': 'inline-block', 'marginLeft': '10px', 'verticalAlign': 'middle'}),
                ], style={'marginTop': '10px'}),
                html.Div([
                    html.Span("RF Confidence: ", style={'fontWeight': 'normal', 'marginTop': '5px', 'display': 'inline-block'}),
                    html.Div([
                        html.Div(f"{rf_conf}%", style={'width': f"{rf_conf}%", 'backgroundColor': '#6f42c1', 'height': '20px', 'lineHeight': '20px', 'color': 'white', 'textAlign': 'right', 'paddingRight': '5px', 'borderRadius': '3px'})
                    ], style={'width': '200px', 'backgroundColor': '#e9ecef', 'borderRadius': '3px', 'display': 'inline-block', 'marginLeft': '10px', 'verticalAlign': 'middle'}),
                ], style={'marginTop': '5px'}),
                html.Div(f"Response Time: {response_time} ms", style={'fontSize': '0.9em', 'color': '#666', 'marginTop': '10px'}),
            ])
            
            result_display = html.Div(result_display_children) # Wrap all children in a single Div

            # SHAP Top Tokens Display
            shap_tokens_display = html.Div([
                html.H4("Top contributing words:", style={'marginTop': '20px', 'marginBottom': '10px', 'color': '#444'}),
                html.Ul([
                    html.Li(f"{word} ({score:.2f})", style={'color': '#dc3545' if score > 0 else '#28a745'})
                    for word, score in shap_words.items()
                ] if shap_words and "error" not in shap_words and "note" not in shap_words else [html.Li(list(shap_words.values())[0]) if shap_words else html.Li("No contributing words found.")])
            ]) if shap_words else html.Div()


            # Feedback section
            feedback_section = html.Div([
                html.H4("Was this prediction correct?", style={'marginTop': '20px', 'marginBottom': '10px', 'color': '#444'}),
                html.Button(
                    'Yes, it was correct',
                    id='feedback-correct-button',
                    n_clicks=0,
                    style={
                        'backgroundColor': '#28a745', 'color': 'white',
                        'border': 'none',
                        'padding': '10px 20px', 'borderRadius': '5px', 'fontSize': '1em',
                        'cursor': 'pointer', 'marginRight': '10px'
                    }
                ),
                html.Button(
                    'No, it was wrong',
                    id='feedback-wrong-button',
                    n_clicks=0,
                    style={
                        'backgroundColor': '#dc3545', 'color': 'white',
                        'border': 'none',
                        'padding': '10px 20px', 'borderRadius': '5px', 'fontSize': '1em',
                        'cursor': 'pointer'
                    }
                ),
                html.Div(id='feedback-confirmation', style={'marginTop': '15px', 'fontWeight': 'bold'})
            ])

            # Store data for feedback callback
            last_prediction_data = {
                'text': user_input,
                'predicted_label': final_label
            }

            return result_display, shap_tokens_display, feedback_section, last_prediction_data, dash.no_update

        except requests.exceptions.RequestException as e:
            return html.Div(), html.Div(), html.Div(), {}, {'type': 'error', 'message': f"Prediction Error: {str(e)}", 'id': n_clicks}
        except Exception as e:
            return html.Div(), html.Div(), html.Div(), {}, {'type': 'error', 'message': f"An unexpected error occurred: {str(e)}", 'id': n_clicks}
    
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


#  Callback for User Feedback 
@app.callback(
    [Output('feedback-confirmation', 'children'),
     Output('notification-trigger', 'data', allow_duplicate=True)],
    [Input('feedback-correct-button', 'n_clicks'),
     Input('feedback-wrong-button', 'n_clicks')],
    [State('last-prediction-data', 'data')],
    prevent_initial_call=True
)
def send_feedback(correct_n, wrong_n, last_prediction_data):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    user_feedback_type = ""
    if button_id == 'feedback-correct-button' and correct_n > 0:
        user_feedback_type = "correct"
    elif button_id == 'feedback-wrong-button' and wrong_n > 0:
        user_feedback_type = "incorrect"
    
    if user_feedback_type and last_prediction_data:
        try:
            response = requests.post(f"{BACKEND_URL}/feedback", json={
                "text": last_prediction_data["text"],
                "predicted_label": last_prediction_data["predicted_label"],
                "user_feedback": user_feedback_type
            })
            response.raise_for_status()
            
            message = f"Thank you for your feedback! ({user_feedback_type})"
            notification_type = 'success'
            notification_id = correct_n + wrong_n
            return html.Div(message, style={'color': '#28a745'}), {'type': notification_type, 'message': message, 'id': notification_id}
        except requests.exceptions.RequestException as e:
            error_message = f"Error sending feedback: {str(e)}"
            notification_id = correct_n + wrong_n
            return html.Div(f"Error: {error_message}", style={'color': 'red'}), {'type': 'error', 'message': error_message, 'id': notification_id}
    
    return dash.no_update, dash.no_update


#  Callback for Notification Display 
@app.callback(
    Output('notification-container', 'children'),
    Input('notification-trigger', 'data'),
    prevent_initial_call=True
)
def display_notification(trigger_data):
    if trigger_data and trigger_data['type']:
        return dcc.Notification(
            id=f'notification-{trigger_data["id"]}',
            title=trigger_data['type'].capitalize(),
            message=trigger_data['message'],
            color='#28a745' if trigger_data['type'] == 'success' else '#dc3545',
            duration=5000,
            dismiss=True,
            is_open=True,
        )
    return html.Div()


@app.callback(
    [
        Output("prediction-count", "figure"),
        Output("language-distribution", "figure"),
        Output("confidence-boxplot", "figure"),
        Output("shap-words", "figure"),
        Output("shap-chart-title", "children"),
        Output("prediction-history-table-container", "children"),
        Output("feedback-summary-chart", "figure"),
        Output("sentiment-distribution", "figure"), # NEW: Output for sentiment chart
        Output("notification-trigger", "data", allow_duplicate=True)
    ],
    [Input("interval-component", "n_intervals")],
    prevent_initial_call=True
)
def update_graphs_and_table(n):
    empty_fig = px.bar(title="No data available or error fetching data", template="plotly_white")
    empty_table_content = html.P("No prediction data available or error fetching data.", style={'padding': '20px', 'color': '#888'})
    
    # MODIFIED: Added sentiment_label and sentiment_score to DataFrame columns
    df = pd.DataFrame(columns=["timestamp", "original_text", "label", "language", "bert_conf", "rf_conf", "top_words", "response_time_ms", "translated_text", "sentiment_label", "sentiment_score"])
    
    latest_input_text_for_shap_title = "Latest Input"
    
    notification_data = dash.no_update

    # Data Fetching 
    try:
        # Fetch monitor logs (which now include sentiment)
        response_logs = requests.get(f"{BACKEND_URL}/get_monitor_logs")
        response_logs.raise_for_status()
        logs = response_logs.json()

        # Fetch monitor stats (which now include sentiment counts)
        response_stats = requests.get(f"{BACKEND_URL}/get_monitor_stats")
        response_stats.raise_for_status()
        stats = response_stats.json()

        if logs:
            df = pd.DataFrame(logs)
            # MODIFIED: Added sentiment_label and sentiment_score to expected columns
            expected_cols = ["timestamp", "original_text", "label", "language", "bert_conf", "rf_conf", "top_words", "response_time_ms", "translated_text", "sentiment_label", "sentiment_score"]
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = None
            
            #  'translated_text' column exists and fill NaNs with empty string for display
            if 'translated_text' not in df.columns:
                df['translated_text'] = None
            df['translated_text'] = df['translated_text'].fillna('')

            # Fill NaN sentiment with "N/A"
            if 'sentiment_label' not in df.columns:
                df['sentiment_label'] = 'N/A'
            df['sentiment_label'] = df['sentiment_label'].fillna('N/A')
            if 'sentiment_score' not in df.columns:
                df['sentiment_score'] = 0.0
            df['sentiment_score'] = df['sentiment_score'].fillna(0.0)

            # Correct the language column in DataFrame for full names
            df['language_full_name'] = df['language'].map(LANGUAGE_NAMES).fillna(df['language'].str.upper())

            if 'original_text' in df.columns and not df['original_text'].empty:
                latest_input_text = df['original_text'].iloc[-1]
                if len(latest_input_text) > 50:
                    latest_input_text_for_shap_title = f"'{latest_input_text[:47]}...'"
                else:
                    latest_input_text_for_shap_title = f"'{latest_input_text}'"
        else:
            print("No recent logs received from /get_monitor_logs endpoint.")

    except requests.exceptions.ConnectionError:
        notification_data = {'type': 'error', 'message': "Error: Could not connect to the Flask backend. Is app.py running?", 'id': n}
        return empty_fig, empty_fig, empty_fig, empty_fig, "SHAP Word-Level Explanations", html.P("Error: Backend connection failed.", style={'padding': '20px', 'color': 'red'}), empty_fig, empty_fig, notification_data
    except requests.exceptions.RequestException as e:
        notification_data = {'type': 'error', 'message': f"Error fetching monitor data: {str(e)}", 'id': n}
        return empty_fig, empty_fig, empty_fig, empty_fig, "SHAP Word-Level Explanations", html.P(f"Error fetching data: {str(e)}", style={'padding': '20px', 'color': 'red'}), empty_fig, empty_fig, notification_data
    except Exception as e:
        notification_data = {'type': 'error', 'message': f"An unexpected error occurred during data processing: {str(e)}", 'id': n}
        return empty_fig, empty_fig, empty_fig, empty_fig, "SHAP Word-Level Explanations", html.P(f"An unexpected error occurred: {str(e)}", style={'padding': '20px', 'color': 'red'}), empty_fig, empty_fig, notification_data

    # --- Graph Generation ---
    fig_prediction_count = empty_fig
    fig_lang_dist = empty_fig
    fig_confidence_boxplot = empty_fig
    fig_shap_words = empty_fig
    fig_feedback_summary = empty_fig
    fig_sentiment_distribution = empty_fig # Initialize sentiment figure
    prediction_history_table = empty_table_content

    if not df.empty:
        # Prediction Count
        label_counts = df["label"].value_counts().reset_index()
        label_counts.columns = ["label", "count"]
        fig_prediction_count = px.bar(label_counts, x="label", y="count", title="Prediction Count",
                                      color="label", color_discrete_map={"AI": "#dc3545", "Human": "#28a745"},
                                      labels={"label": "Predicted Label", "count": "Count"},
                                      template="plotly_white")
        fig_prediction_count.update_layout(showlegend=False, xaxis_title="", yaxis_title="Count")

        # Language Distribution
        lang_counts = df["language_full_name"].value_counts().reset_index()
        lang_counts.columns = ["language", "count"]
        fig_lang_dist = px.pie(lang_counts, names="language", values="count", title="Language Distribution",
                               labels={"language": "Detected Language", "count": "Count"},
                               template="plotly_white")
        fig_lang_dist.update_traces(textposition='inside', textinfo='percent+label')

        # BERT Confidence Boxplot
        fig_confidence_boxplot = px.box(df, x="label", y="bert_conf", title="BERT Confidence Score by Label",
                                         color="label", color_discrete_map={"AI": "#dc3545", "Human": "#28a745"},
                                         labels={"label": "Predicted Label", "bert_conf": "BERT Confidence (%)"},
                                         template="plotly_white", points="all")
        fig_confidence_boxplot.update_layout(showlegend=False, xaxis_title="", yaxis_title="Confidence (%)")

        # SHAP Word-Level Explanations (for the *last* prediction)
        if 'top_words' in df.columns and len(df['top_words']) > 0 and isinstance(df['top_words'].iloc[-1], dict) and "error" not in df['top_words'].iloc[-1] and "note" not in df['top_words'].iloc[-1]:
            last_shap_words = df['top_words'].iloc[-1]
            if last_shap_words:
                shap_df = pd.DataFrame(list(last_shap_words.items()), columns=['word', 'importance'])
                shap_df['impact_direction'] = shap_df['importance'].apply(lambda x: 'Towards AI' if x > 0 else 'Towards Human')
                shap_df['abs_importance'] = shap_df['importance'].abs()
                shap_df = shap_df.sort_values(by='abs_importance', ascending=False)
                
                fig_shap_words = px.bar(shap_df, x='word', y='importance',
                                         title=f"SHAP Token Importances (Green: Human, Red: AI)",
                                         color='impact_direction',
                                         color_discrete_map={'Towards AI': '#dc3545', 'Towards Human': '#28a745'},
                                         labels={'importance': 'SHAP Importance', 'word': 'Word'},
                                         template="plotly_white")
                fig_shap_words.update_layout(showlegend=True, xaxis_title="Word", yaxis_title="SHAP Importance")
            else:
                fig_shap_words = px.bar(title="No SHAP data for this input.", template="plotly_white")
        else:
            fig_shap_words = px.bar(title=df['top_words'].iloc[-1].get("note", "No SHAP data available or error."), template="plotly_white")
            if "error" in df['top_words'].iloc[-1]:
                notification_data = {'type': 'warning', 'message': f"SHAP calculation error: {df['top_words'].iloc[-1]['error']}", 'id': n}


        # Prediction History Table
        # MODIFIED: Added "Sentiment Label" and "Sentiment Score" columns
        display_df = df[["timestamp", "original_text", "label", "language_full_name", "bert_conf", "rf_conf", "response_time_ms", "translated_text", "sentiment_label", "sentiment_score"]].copy()
        display_df.columns = ["Timestamp", "Input Text", "Predicted Label", "Language", "BERT Conf (%)", "RF Conf (%)", "Response Time (ms)", "Translated Text (if not English)", "Sentiment Label", "Sentiment Score (%)"]
        
        prediction_history_table = dash_table.DataTable(
            id='table-prediction-history',
            columns=[{"name": i, "id": i} for i in display_df.columns],
            data=display_df.to_dict('records'),
            style_table={'overflowX': 'auto', 'minWidth': '100%'},
            style_header={
                'backgroundColor': '#f0f0f0',
                'fontWeight': 'bold',
                'textAlign': 'left',
                'padding': '12px 15px',
                'borderBottom': '2px solid #ddd'
            },
            style_data={
                'whiteSpace': 'normal',
                'height': 'auto',
                'textAlign': 'left',
                'padding': '10px 15px',
                'borderBottom': '1px solid #eee'
            },
            style_cell_conditional=[
                {'if': {'column_id': 'Timestamp'}, 'width': '10%'}, # Adjusted widths
                {'if': {'column_id': 'Input Text'}, 'width': '20%'},
                {'if': {'column_id': 'Translated Text (if not English)'}, 'width': '20%', 'fontStyle': 'italic', 'color': '#666'},
                {'if': {'column_id': 'Predicted Label'}, 'width': '8%'},
                {'if': {'column_id': 'Language'}, 'width': '7%'},
                {'if': {'column_id': 'BERT Conf (%)'}, 'width': '7%'},
                {'if': {'column_id': 'RF Conf (%)'}, 'width': '7%'},
                {'if': {'column_id': 'Response Time (ms)'}, 'width': '8%'},
                {'if': {'column_id': 'Sentiment Label'}, 'width': '7%'}, # NEW: Style for sentiment label
                {'if': {'column_id': 'Sentiment Score (%)'}, 'width': '7%'}, # NEW: Style for sentiment score
            ]
        )

    # Feedback Summary Chart
    feedback_response = requests.get(f"{BACKEND_URL}/feedback_summary")
    feedback_response.raise_for_status()
    feedback_data = feedback_response.json().get("feedback_data", [])
    
    if feedback_data:
        feedback_df = pd.DataFrame(feedback_data)
        feedback_summary = feedback_df['user_feedback'].value_counts().reset_index()
        feedback_summary.columns = ['Feedback Type', 'Count']
        fig_feedback_summary = px.pie(feedback_summary, names='Feedback Type', values='Count',
                                       title="User Feedback Summary",
                                       color='Feedback Type',
                                       color_discrete_map={'correct': '#28a745', 'incorrect': '#dc3545'},
                                       template="plotly_white")
        fig_feedback_summary.update_traces(textposition='inside', textinfo='percent+label')
    else:
        fig_feedback_summary = px.pie(title="No user feedback yet.", template="plotly_white")

    # NEW: Sentiment Distribution Chart
    # Use the 'stats' dictionary fetched from the backend for the sentiment counts
    sentiment_counts = {
        'POSITIVE': stats.get('sentiment_positive', 0),
        'NEGATIVE': stats.get('sentiment_negative', 0),
        'NEUTRAL': stats.get('sentiment_neutral', 0),
        'N/A': stats.get('sentiment_n/a', 0),
        'Error': stats.get('sentiment_error', 0)
    }

    labels = [k for k, v in sentiment_counts.items() if v > 0]
    values = [v for k, v in sentiment_counts.items() if v > 0]

    if values:
        sentiment_colors = {'POSITIVE': '#28a745', 'NEGATIVE': '#dc3545', 'NEUTRAL': '#ffc107', 'N/A': '#6c757d', 'Error': '#6c757d'}
        
        fig_sentiment_distribution = px.pie(names=labels, values=values, title="Sentiment Distribution",
                                            color=labels, color_discrete_map=sentiment_colors,
                                            labels={"names": "Sentiment", "values": "Count"},
                                            template="plotly_white")
        fig_sentiment_distribution.update_traces(textposition='inside', textinfo='percent+label')
    else:
        fig_sentiment_distribution = px.pie(title="No sentiment data available.", template="plotly_white")


    return fig_prediction_count, fig_lang_dist, fig_confidence_boxplot, fig_shap_words, f"SHAP Word-Level Explanations for {latest_input_text_for_shap_title}", prediction_history_table, fig_feedback_summary, fig_sentiment_distribution, notification_data


# Download Feedback Logs Callback
@app.callback(
    Output("download-feedback-json", "data"),
    Input("download-feedback-button", "n_clicks"),
    prevent_initial_call=True
)
def download_feedback_logs_file(n_clicks):
    if n_clicks > 0:
        try:
            response = requests.get(f"{BACKEND_URL}/download_feedback_logs")
            response.raise_for_status()
            feedback_json_data = response.json()
            
            return dcc.send_string(json.dumps(feedback_json_data, indent=4), "feedback_logs.json")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading feedback logs: {e}")
            return None
    return None

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
