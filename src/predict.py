import os
from src.preprocessing import clean_text

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



def load_models():
    """
    Load all trained components required
    for hybrid phishing prediction.
    """

    print("Loading hybrid phishing detection models...")

    lstm_model = load_model(
        os.path.join(
            MODELS_DIR,
            MODEL_NAME,
        )
    )

    tokenizer = load_pickle(
        os.path.join(
            MODELS_DIR,
            "tokenizer.pkl",
        )
    )

    risk_model = load_pickle(
        os.path.join(
            MODELS_DIR,
            SENTIMENT_MODEL_NAME,
        )
    )

    vectorizer = load_pickle(
        os.path.join(
            MODELS_DIR,
            TFIDF_VECTORIZER_NAME,
        )
    )

    print("Models loaded successfully.")

    return (
        lstm_model,
        tokenizer,
        risk_model,
        vectorizer,
    )


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

    return float(probability)


def get_risk_probability(
    text,
    risk_model,
    vectorizer,
):
    """
    Generate P2:
    Naive Bayes social-engineering risk probability.
    """

    tfidf_features = vectorizer.transform(
        [text]
    )

    probability = risk_model.predict_proba(
        tfidf_features
    )[0][1]

    return float(probability)


def calculate_hybrid_score(
    p1,
    p2,
):
    """
    Calculate final weighted hybrid probability.

    P_final = 0.60(P1) + 0.40(P2)
    """

    final_score = (
        LSTM_WEIGHT * p1
        + RISK_WEIGHT * p2
    )

    return float(final_score)


def get_risk_rating(score):
    """
    Convert the final probability into a
    human-readable risk rating.

    Note:
    These ranges are presentation categories,
    not separately trained classification thresholds.
    """

    if score < 0.25:
        return "LOW"

    elif score < HYBRID_THRESHOLD:
        return "MODERATE"

    elif score < 0.75:
        return "HIGH"

    else:
        return "VERY HIGH"


def predict_email(
    email_text,
    lstm_model,
    tokenizer,
    risk_model,
    vectorizer,
):
    """
    Run the complete hybrid phishing
    prediction pipeline.
    """

    cleaned_text = clean_text(
    email_text
   )

    if not cleaned_text:
        raise ValueError(
            "Email text is empty after preprocessing."
        )

    # LSTM probability
    p1 = get_lstm_probability(
        cleaned_text,
        lstm_model,
        tokenizer,
    )

    # Social-engineering risk probability
    p2 = get_risk_probability(
        cleaned_text,
        risk_model,
        vectorizer,
    )

    # Hybrid probability
    final_score = calculate_hybrid_score(
        p1,
        p2,
    )

    if final_score >= HYBRID_THRESHOLD:
        prediction = "PHISHING"
    else:
        prediction = "LEGITIMATE"

    risk_rating = get_risk_rating(
        final_score
    )

    return {
        "lstm_probability": p1,
        "risk_probability": p2,
        "hybrid_score": final_score,
        "prediction": prediction,
        "risk_rating": risk_rating,
    }


def display_result(result):
    """
    Display prediction results.
    """

    print("\n" + "=" * 55)

    print(
        "HYBRID PHISHING EMAIL DETECTION RESULT"
    )

    print("=" * 55)

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

    print(
        f"\nPrediction                     : "
        f"{result['prediction']}"
    )

    print(
        f"Risk Rating                    : "
        f"{result['risk_rating']}"
    )

    print("\n" + "=" * 55)


def main():
    """
    Interactive command-line prediction interface.
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

        lines.append(line)

    email_text = " ".join(lines)

    if not email_text.strip():

        print(
            "\nNo email text was entered."
        )

        return

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


if __name__ == "__main__":
    main()