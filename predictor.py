import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from langdetect import detect
import joblib
from collections import defaultdict
from datetime import datetime
import shap
import os
import json
import time
from googletrans import Translator

# Initialize
translator = Translator()
bert_model_path = "bert_finetunedfixed_model"
tokenizer = BertTokenizer.from_pretrained(bert_model_path)
bert_model = BertForSequenceClassification.from_pretrained(bert_model_path)
bert_model.eval()
rf_model = joblib.load("bert_rf_cls_final.joblib")

try:
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    print("Sentiment analysis initialized.")
except Exception as e:
    print(f"Sentiment pipeline error: {e}")
    sentiment_pipeline = None

bert_pipeline = pipeline("text-classification", model=bert_model, tokenizer=tokenizer, return_all_scores=True)
try:
    explainer = shap.Explainer(bert_pipeline, output_idx=1, nsamples=100)
    print("SHAP initialized.")
except Exception as e:
    print(f"SHAP explainer init failed: {e}")
    explainer = None

monitor_stats = defaultdict(int)
monitor_logs = []
feedback_logs_file = "feedback_logs.json"
feedback_logs = []

if os.path.exists(feedback_logs_file):
    try:
        with open(feedback_logs_file, 'r') as f:
            feedback_logs = json.load(f)
    except json.JSONDecodeError:
        feedback_logs = []

LANGUAGE_NAMES = {
    'en': 'English', 'hi': 'Hindi', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
    'zh-cn': 'Chinese (Simplified)', 'ja': 'Japanese', 'ru': 'Russian', 'pt': 'Portuguese',
    'it': 'Italian', 'ko': 'Korean', 'ar': 'Arabic', 'bn': 'Bengali', 'pa': 'Punjabi',
    'gu': 'Gujarati', 'ml': 'Malayalam', 'te': 'Telugu', 'ta': 'Tamil', 'ur': 'Urdu',
    'tr': 'Turkish', 'pl': 'Polish', 'nl': 'Dutch', 'sv': 'Swedish', 'fi': 'Finnish',
    'no': 'Norwegian', 'da': 'Danish', 'id': 'Indonesian', 'ms': 'Malay', 'th': 'Thai',
    'vi': 'Vietnamese', 'he': 'Hebrew', 'el': 'Greek', 'cs': 'Czech', 'hu': 'Hungarian',
    'ro': 'Romanian', 'uk': 'Ukrainian', 'fa': 'Persian', 'bg': 'Bulgarian', 'sk': 'Slovak',
    'sl': 'Slovenian', 'sr': 'Serbian', 'hr': 'Croatian', 'lt': 'Lithuanian', 'lv': 'Latvian',
    'et': 'Estonian', 'is': 'Icelandic', 'ga': 'Irish', 'cy': 'Welsh', 'mt': 'Maltese',
    'sq': 'Albanian', 'mk': 'Macedonian', 'tl': 'Tagalog', 'sw': 'Swahili',
}


def log_monitor_data(text, label, lang, bert_conf, rf_conf, shap_words, response_ms, translated=None, sentiment_label=None, sentiment_score=None):
    monitor_stats[f"count_{label}"] += 1
    monitor_stats[f"lang_{lang}"] += 1
    if sentiment_label:
        monitor_stats[f"sentiment_{sentiment_label.lower()}"] += 1

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "original_text": text,
        "label": label,
        "language": lang,
        "bert_conf": bert_conf,
        "rf_conf": rf_conf,
        "top_words": shap_words,
        "response_time_ms": response_ms
    }
    if translated:
        log_entry["translated_text"] = translated
    if sentiment_label:
        log_entry["sentiment_label"] = sentiment_label
        log_entry["sentiment_score"] = sentiment_score

    monitor_logs.append(log_entry)
    if len(monitor_logs) > 1000:
        monitor_logs.pop(0)


def log_feedback(text, predicted_label, user_feedback):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "text": text,
        "predicted_label": predicted_label,
        "user_feedback": user_feedback
    }
    feedback_logs.append(entry)
    with open(feedback_logs_file, 'w') as f:
        json.dump(feedback_logs, f, indent=4)


def predict_batch(texts, prediction_strategy="ensemble"):
    results = []
    for text in texts:
        start_time = time.time()
        lang = detect(text)
        translated = None

        # Translate if needed
        if lang != "en":
            try:
                translated = translator.translate(text, src=lang, dest="en").text
            except Exception:
                translated = "Translation unavailable."

        text_for_model = translated if translated and translated != "Translation unavailable." else text

        # Sentiment
        sentiment_label = "N/A"
        sentiment_score = 0.0
        if sentiment_pipeline and text_for_model:
            try:
                result = sentiment_pipeline(text_for_model)[0]
                sentiment_label = result["label"]
                sentiment_score = round(result["score"] * 100, 2)
            except Exception:
                sentiment_label = "Error"

        # Prediction
        inputs = tokenizer(text_for_model, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        bert_model.to(device)

        with torch.no_grad():
            logits = bert_model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = np.argmax(probs)
            conf_bert = round(probs[pred_idx] * 100, 2)
            label_bert = "AI" if pred_idx == 1 else "Human"

            cls_embedding = bert_model.bert(**inputs).last_hidden_state[:, 0, :].cpu().numpy()
            pred_rf = rf_model.predict(cls_embedding)[0]
            conf_rf = round(np.max(rf_model.predict_proba(cls_embedding)) * 100, 2)
            label_rf = "AI" if pred_rf == 1 else "Human"

        # Final label logic
        if prediction_strategy == "bert":
            final_label = label_bert
        elif prediction_strategy == "rf":
            final_label = label_rf
        else:
            final_label = label_bert if conf_bert >= 85 else label_rf

        # SHAP
        shap_words = {}
        if explainer and translated != "Translation unavailable.":
            try:
                shap_vals = explainer([text_for_model])[0]
                token_scores = {}
                for i, token in enumerate(shap_vals.data):
                    val = shap_vals.values[i][1] if isinstance(shap_vals.values[i], np.ndarray) else shap_vals.values[i]
                    token_scores[token.replace("##", "")] = round(float(val), 4)
                shap_words = dict(sorted(token_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
            except Exception:
                shap_words = {"error": "SHAP generation failed"}
        else:
            shap_words = {"note": "SHAP not calculated for non-English or missing explainer"}

        response_time = round((time.time() - start_time) * 1000, 2)

        log_monitor_data(text, final_label, lang, conf_bert, conf_rf, shap_words, response_time, translated, sentiment_label, sentiment_score)

        results.append({
            "Original Text": text,
            "Language": lang,
            "Translated Text": translated,
            "BERT Label": label_bert,
            "BERT Conf": conf_bert,
            "RF Label": label_rf,
            "RF Conf": conf_rf,
            "Final Label": final_label,
            "SHAP Words": shap_words,
            "Response Time (ms)": response_time,
            "Sentiment Label": sentiment_label,
            "Sentiment Score": sentiment_score
        })

    return results
