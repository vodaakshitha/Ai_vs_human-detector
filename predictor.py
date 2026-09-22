import os
import json
import time
from collections import defaultdict
from datetime import datetime

import torch
import numpy as np
import joblib

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
)

from langdetect import detect
from googletrans import Translator

from constants import LANGUAGE_NAMES


# ============================================================
# MEMORY OPTIMIZATION
# ============================================================

# Render Free has limited memory.
# Limit PyTorch CPU thread usage.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

DEVICE = torch.device("cpu")

# Hugging Face model
bert_model_path = os.getenv(
    "BERT_MODEL_PATH",
    "akshithavoda/bert-finetuned-ai-human"
)


# ============================================================
# TRANSLATOR
# ============================================================

translator = Translator()


# ============================================================
# BERT MODEL
# ============================================================

print("Loading BERT tokenizer...")

tokenizer = BertTokenizer.from_pretrained(
    bert_model_path
)

print("Loading BERT model...")

bert_model = BertForSequenceClassification.from_pretrained(
    bert_model_path
)

bert_model.to(DEVICE)
bert_model.eval()

print("BERT model loaded successfully.")


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

print("Loading Random Forest model...")

rf_model = joblib.load("bert_rf_cls_final.joblib")

print("Random Forest model loaded successfully.")


# ============================================================
# OPTIONAL FEATURES
# ============================================================

# These are disabled by default to reduce memory usage.
#
# The original application loaded:
#   1. DistilBERT sentiment model
#   2. SHAP explainer
#
# Both can consume substantial memory.
#
# Set these environment variables to "true" later if you
# intentionally want to enable them.

ENABLE_SENTIMENT = (
    os.getenv("ENABLE_SENTIMENT", "false").lower() == "true"
)

ENABLE_SHAP = (
    os.getenv("ENABLE_SHAP", "false").lower() == "true"
)


sentiment_pipeline = None
explainer = None


# ============================================================
# OPTIONAL SENTIMENT INITIALIZATION
# ============================================================

if ENABLE_SENTIMENT:
    try:
        from transformers import pipeline

        print("Loading sentiment model...")

        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1
        )

        print("Sentiment analysis initialized.")

    except Exception as e:
        print(f"Sentiment pipeline error: {e}")
        sentiment_pipeline = None


# ============================================================
# OPTIONAL SHAP INITIALIZATION
# ============================================================

if ENABLE_SHAP:
    try:
        import shap

        print("Loading SHAP explainer...")

        bert_pipeline = pipeline(
            "text-classification",
            model=bert_model,
            tokenizer=tokenizer,
            top_k=None
        )

        explainer = shap.Explainer(
            bert_pipeline,
            output_idx=1,
            nsamples=100
        )

        print("SHAP initialized.")

    except Exception as e:
        print(f"SHAP explainer initialization failed: {e}")
        explainer = None


# ============================================================
# MONITORING
# ============================================================

monitor_stats = defaultdict(int)

monitor_logs = []

feedback_logs_file = "feedback_logs.json"

feedback_logs = []


# Load existing feedback logs if available
if os.path.exists(feedback_logs_file):
    try:
        with open(feedback_logs_file, "r") as f:
            feedback_logs = json.load(f)

    except json.JSONDecodeError:
        feedback_logs = []


# ============================================================
# MONITORING FUNCTION
# ============================================================

def log_monitor_data(
    text,
    label,
    lang,
    bert_conf,
    rf_conf,
    shap_words,
    response_ms,
    translated=None,
    sentiment_label=None,
    sentiment_score=None
):

    monitor_stats[f"count_{label}"] += 1
    monitor_stats[f"lang_{lang}"] += 1

    if sentiment_label:
        monitor_stats[
            f"sentiment_{sentiment_label.lower()}"
        ] += 1

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

    # Keep only latest 1000 records
    if len(monitor_logs) > 1000:
        monitor_logs.pop(0)


# ============================================================
# FEEDBACK FUNCTION
# ============================================================

def log_feedback(
    text,
    predicted_label,
    user_feedback
):

    entry = {
        "timestamp": datetime.now().isoformat(),
        "text": text,
        "predicted_label": predicted_label,
        "user_feedback": user_feedback
    }

    feedback_logs.append(entry)

    with open(feedback_logs_file, "w") as f:
        json.dump(
            feedback_logs,
            f,
            indent=4
        )


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_batch(
    texts,
    prediction_strategy="ensemble"
):

    results = []

    for text in texts:

        start_time = time.time()

        # ----------------------------------------------------
        # Language detection
        # ----------------------------------------------------

        lang = detect(text)

        translated = None


        # ----------------------------------------------------
        # Translation
        # ----------------------------------------------------

        if lang != "en":

            try:

                translated = translator.translate(
                    text,
                    src=lang,
                    dest="en"
                ).text

            except Exception:

                translated = "Translation unavailable."


        text_for_model = (
            translated
            if translated
            and translated != "Translation unavailable."
            else text
        )


        # ----------------------------------------------------
        # Sentiment
        # ----------------------------------------------------

        sentiment_label = "N/A"
        sentiment_score = 0.0

        if sentiment_pipeline and text_for_model:

            try:

                result = sentiment_pipeline(
                    text_for_model
                )[0]

                sentiment_label = result["label"]

                sentiment_score = round(
                    result["score"] * 100,
                    2
                )

            except Exception:

                sentiment_label = "Error"


        # ----------------------------------------------------
        # BERT TOKENIZATION
        # ----------------------------------------------------

        inputs = tokenizer(
            text_for_model,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }


        # ----------------------------------------------------
        # BERT + RANDOM FOREST
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = bert_model(
                **inputs
            )

            logits = outputs.logits

            probs = torch.softmax(
                logits,
                dim=1
            )[0].cpu().numpy()

            pred_idx = int(
                np.argmax(probs)
            )

            conf_bert = round(
                float(probs[pred_idx]) * 100,
                2
            )

            label_bert = (
                "AI"
                if pred_idx == 1
                else "Human"
            )


            # ------------------------------------------------
            # CLS EMBEDDING FOR RANDOM FOREST
            # ------------------------------------------------

            cls_embedding = (
                bert_model.bert(
                    **inputs
                )
                .last_hidden_state[:, 0, :]
                .cpu()
                .numpy()
            )

            pred_rf = rf_model.predict(
                cls_embedding
            )[0]

            conf_rf = round(
                float(
                    np.max(
                        rf_model.predict_proba(
                            cls_embedding
                        )
                    )
                ) * 100,
                2
            )

            label_rf = (
                "AI"
                if pred_rf == 1
                else "Human"
            )


        # ----------------------------------------------------
        # FINAL PREDICTION
        # ----------------------------------------------------

        if prediction_strategy == "bert":

            final_label = label_bert

        elif prediction_strategy == "rf":

            final_label = label_rf

        else:

            final_label = (
                label_bert
                if conf_bert >= 85
                else label_rf
            )


        # ----------------------------------------------------
        # SHAP
        # ----------------------------------------------------

        shap_words = {}

        if explainer:

            try:

                shap_vals = explainer(
                    [text_for_model]
                )[0]

                token_scores = {}

                for i, token in enumerate(
                    shap_vals.data
                ):

                    val = (
                        shap_vals.values[i][1]
                        if isinstance(
                            shap_vals.values[i],
                            np.ndarray
                        )
                        else shap_vals.values[i]
                    )

                    token_scores[
                        token.replace("##", "")
                    ] = round(
                        float(val),
                        4
                    )

                shap_words = dict(
                    sorted(
                        token_scores.items(),
                        key=lambda x: abs(x[1]),
                        reverse=True
                    )[:5]
                )

            except Exception:

                shap_words = {
                    "error": "SHAP generation failed"
                }

        else:

            shap_words = {
                "note": "SHAP disabled to reduce memory usage"
            }


        # ----------------------------------------------------
        # RESPONSE TIME
        # ----------------------------------------------------

        response_time = round(
            (time.time() - start_time) * 1000,
            2
        )


        # ----------------------------------------------------
        # MONITORING
        # ----------------------------------------------------

        log_monitor_data(
            text,
            final_label,
            lang,
            conf_bert,
            conf_rf,
            shap_words,
            response_time,
            translated,
            sentiment_label,
            sentiment_score
        )


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

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
