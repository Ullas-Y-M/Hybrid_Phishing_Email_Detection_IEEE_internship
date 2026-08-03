import os

import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import (
    MODELS_DIR,
    MODEL_NAME,
    SENTIMENT_MODEL_NAME,
    TFIDF_VECTORIZER_NAME,
    MAX_SEQUENCE_LENGTH,
    HYBRID_THRESHOLD,
    LSTM_WEIGHT,
    RISK_WEIGHT,
)

from src.utils import load_pickle

# Use the exact same cleaning function used during training
from src.preprocessing import clean_text

# Phase 9 explainability functions
from src.explainability import (
    detect_risk_indicators,
    create_explanation,
    count_detected_indicators,
    display_explanation,
)


# --------------------------------------------------
# Load Models
# --------------------------------------------------

def load_models():
    """
    Load all trained components required for
    hybrid phishing prediction.
    """

    print(
        "Loading hybrid phishing detection models..."
    )

    # Load LSTM model
    lstm_model = load_model(
        os.path.join(
            MODELS_DIR,
            MODEL_NAME,
        )
    )

    # Load tokenizer
    tokenizer = load_pickle(
        os.path.join(
            MODELS_DIR,
            "tokenizer.pkl",
        )
    )

    # Load sentiment-aware Naive Bayes model
    risk_model = load_pickle(
        os.path.join(
            MODELS_DIR,
            SENTIMENT_MODEL_NAME,
        )
    )

    # Load TF-IDF vectorizer
    vectorizer = load_pickle(
        os.path.join(
            MODELS_DIR,
            TFIDF_VECTORIZER_NAME,
        )
    )

    print(
        "Models loaded successfully."
    )

    return (
        lstm_model,
        tokenizer,
        risk_model,
        vectorizer,
    )


# --------------------------------------------------
# LSTM Probability
# --------------------------------------------------

def get_lstm_probability(
    text,
    lstm_model,
    tokenizer,
):
    """
    Generate P1:
    LSTM phishing probability.
    """

    sequence = tokenizer.texts_to_sequences(
        [text]
    )

    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
    )

    probability = lstm_model.predict(
        padded_sequence,
        verbose=0,
    )[0][0]

    return float(
        probability
    )


# --------------------------------------------------
# Risk Analyzer Probability
# --------------------------------------------------

def get_risk_probability(
    text,
    risk_model,
    vectorizer,
):
    """
    Generate P2:
    Sentiment-aware social-engineering
    risk probability.
    """

    tfidf_features = vectorizer.transform(
        [text]
    )

    probability = risk_model.predict_proba(
        tfidf_features
    )[0][1]

    return float(
        probability
    )


# --------------------------------------------------
# Hybrid Probability
# --------------------------------------------------

def calculate_hybrid_score(
    p1,
    p2,
):
    """
    Calculate the final hybrid phishing score.

    P_final =
        LSTM_WEIGHT * P1
        +
        RISK_WEIGHT * P2

    Current validated configuration:

        P_final = 0.60(P1) + 0.40(P2)
    """

    final_score = (
        LSTM_WEIGHT * p1
        + RISK_WEIGHT * p2
    )

    return float(
        final_score
    )


# --------------------------------------------------
# Risk Rating
# --------------------------------------------------

def get_risk_rating(score):
    """
    Convert the final hybrid probability into
    a human-readable display category.

    These categories are for presentation only.
    They are not separately trained risk classes.
    """

    if score < 0.25:
        return "LOW"

    elif score < HYBRID_THRESHOLD:
        return "MODERATE"

    elif score < 0.75:
        return "HIGH"

    else:
        return "VERY HIGH"


# --------------------------------------------------
# Complete Email Prediction
# --------------------------------------------------

def predict_email(
    email_text,
    lstm_model,
    tokenizer,
    risk_model,
    vectorizer,
):
    """
    Run the complete hybrid phishing prediction
    and explainability pipeline.
    """

    # ------------------------------------------
    # Preprocessing
    # ------------------------------------------

    cleaned_text = clean_text(
        email_text
    )

    if not cleaned_text:

        raise ValueError(
            "Email text is empty after preprocessing."
        )

    # ------------------------------------------
    # P1 - LSTM Probability
    # ------------------------------------------

    p1 = get_lstm_probability(
        cleaned_text,
        lstm_model,
        tokenizer,
    )

    # ------------------------------------------
    # P2 - Social Engineering Risk Probability
    # ------------------------------------------

    p2 = get_risk_probability(
        cleaned_text,
        risk_model,
        vectorizer,
    )

    # ------------------------------------------
    # Hybrid Fusion
    # ------------------------------------------

    final_score = calculate_hybrid_score(
        p1,
        p2,
    )

    # ------------------------------------------
    # Final Classification
    # ------------------------------------------

    if final_score >= HYBRID_THRESHOLD:

        prediction = "PHISHING"

    else:

        prediction = "LEGITIMATE"

    # ------------------------------------------
    # Risk Rating
    # ------------------------------------------

    risk_rating = get_risk_rating(
        final_score
    )

    # ------------------------------------------
    # Phase 9 Explainability
    # ------------------------------------------

    # Use raw email text here so that human-readable
    # indicators can be detected before cleaning
    # removes potentially useful information.

    indicators = detect_risk_indicators(
        email_text
    )

    indicator_count = count_detected_indicators(
        indicators
    )

    explanation = create_explanation(
        indicators
    )

    # ------------------------------------------
    # Return Complete Result
    # ------------------------------------------

    return {
        "lstm_probability": p1,
        "risk_probability": p2,
        "hybrid_score": final_score,
        "prediction": prediction,
        "risk_rating": risk_rating,
        "indicator_count": indicator_count,
        "indicators": indicators,
        "explanation": explanation,
    }


# --------------------------------------------------
# Display Prediction
# --------------------------------------------------

def display_result(result):
    """
    Display prediction and explainability results.
    """

    print(
        "\n" + "=" * 60
    )

    print(
        "HYBRID PHISHING EMAIL DETECTION RESULT"
    )

    print(
        "=" * 60
    )

    # ------------------------------------------
    # Model Probabilities
    # ------------------------------------------

    print(
        f"\nLSTM Phishing Probability (P1) : "
        f"{result['lstm_probability']:.4f}"
    )

    print(
        f"Risk Probability (P2)          : "
        f"{result['risk_probability']:.4f}"
    )

    print(
        f"Hybrid Score                   : "
        f"{result['hybrid_score']:.4f}"
    )

    # ------------------------------------------
    # Final Decision
    # ------------------------------------------

    print(
        f"\nPrediction                     : "
        f"{result['prediction']}"
    )

    print(
        f"Risk Rating                    : "
        f"{result['risk_rating']}"
    )

    # ------------------------------------------
    # Explainability
    # ------------------------------------------

    print(
        f"Detected Indicator Categories  : "
        f"{result['indicator_count']}"
    )

    display_explanation(
        result["indicators"]
    )

    print(
        f"\nExplanation: "
        f"{result['explanation']}"
    )

    print(
        "\n" + "=" * 60
    )


# --------------------------------------------------
# Interactive CLI
# --------------------------------------------------

def main():
    """
    Interactive command-line interface for
    hybrid phishing email detection.
    """

    (
        lstm_model,
        tokenizer,
        risk_model,
        vectorizer,
    ) = load_models()

    print(
        "\nHybrid Phishing Email Detector"
    )

    print(
        "Enter the email you want to analyze."
    )

    print(
        "Press ENTER twice when finished.\n"
    )

    lines = []

    while True:

        line = input()

        if line == "":
            break

        lines.append(
            line
        )

    email_text = " ".join(
        lines
    )

    if not email_text.strip():

        print(
            "\nNo email text was entered."
        )

        return

    try:

        result = predict_email(
            email_text,
            lstm_model,
            tokenizer,
            risk_model,
            vectorizer,
        )

        display_result(
            result
        )

    except Exception as error:

        print(
            f"\nPrediction failed: {error}"
        )


# --------------------------------------------------
# Entry Point
# --------------------------------------------------

if __name__ == "__main__":
    main()