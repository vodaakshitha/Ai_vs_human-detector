from flask import Flask, request, jsonify, send_file # Import send_file
import os # Import os for file path checks
from predictor import predict_batch, monitor_stats, monitor_logs, log_feedback, feedback_logs

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    texts = data.get("texts", [])
    prediction_strategy = data.get("strategy", "ensemble")

    predictions = predict_batch(texts, prediction_strategy)
    return jsonify(predictions)

@app.route("/monitor", methods=["GET"])
def monitor():
    # This endpoint now returns aggregated summary and recent logs
    return jsonify({
        "summary": dict(monitor_stats),
        "recent_logs": monitor_logs[-50:] # Limit to last 50 logs for performance
    })

# NEW: Endpoint to get full monitor stats for analytics graphs
@app.route("/get_monitor_stats", methods=["GET"])
def get_monitor_stats():
    return jsonify(dict(monitor_stats))

# NEW: Endpoint to get all recent logs for the history table
@app.route("/get_monitor_logs", methods=["GET"])
def get_monitor_logs():
    return jsonify(monitor_logs) # Return all logs for table processing on client side

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json()
    text = data.get("text")
    predicted_label = data.get("predicted_label")
    user_feedback = data.get("user_feedback")

    if not all([text, predicted_label, user_feedback]):
        return jsonify({"status": "error", "message": "Missing required feedback data."}), 400

    log_feedback(text, predicted_label, user_feedback)

    return jsonify({"status": "success", "message": "Feedback received!"}), 200

@app.route("/feedback_summary", methods=["GET"])
def feedback_summary():
    return jsonify({"feedback_data": feedback_logs})

@app.route("/download_feedback_logs", methods=["GET"])
def download_feedback_logs():
    if os.path.exists("feedback_logs.json"):
        return send_file("feedback_logs.json", as_attachment=True, download_name="feedback_logs.json", mimetype="application/json")
    return jsonify({"status": "error", "message": "Feedback logs file not found."}), 404


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
