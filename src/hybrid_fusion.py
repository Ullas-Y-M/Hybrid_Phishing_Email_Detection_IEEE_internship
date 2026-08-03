import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from tensorflow.keras.models import load_model

from config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    EVALUATION_DIR,
    MODEL_NAME,
    SENTIMENT_MODEL_NAME,
    TFIDF_VECTORIZER_NAME,
    TEST_SIZE,
    RANDOM_STATE,
    HYBRID_THRESHOLD,
)

from src.utils import (
    create_directory,
    load_pickle,
    save_json,
)


def load_resources():
    """
    Load dataset, LSTM model, Naive Bayes model,
    TF-IDF vectorizer and LSTM test sequences.
    """

    print("Loading hybrid model resources...")

    dataset = pd.read_csv(
        os.path.join(
            PROCESSED_DATA_DIR,
            "sentiment_risk_dataset.csv",
        )
    )

    X_test_lstm = np.load(
        os.path.join(
            PROCESSED_DATA_DIR,
            "X_test.npy",
        )
    )

    y_test_lstm = np.load(
        os.path.join(
            PROCESSED_DATA_DIR,
            "y_test.npy",
        )
    )

    lstm_model = load_model(
        os.path.join(
            MODELS_DIR,
            MODEL_NAME,
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

    return (
        dataset,
        X_test_lstm,
        y_test_lstm,
        lstm_model,
        risk_model,
        vectorizer,
    )


def prepare_text_test_data(dataset):
    """
    Reproduce the same test split used by the LSTM
    tokenizer so that P1 and P2 correspond to the
    same emails.
    """

    X = dataset["text"]
    y = dataset["label"]

    _, X_test_text, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_test_text, y_test.to_numpy()


def generate_probabilities(
    lstm_model,
    risk_model,
    vectorizer,
    X_test_lstm,
    X_test_text,
):
    """
    Generate P1 and P2 probabilities.
    """

    print("\nGenerating LSTM probabilities (P1)...")

    p1 = lstm_model.predict(
        X_test_lstm,
        verbose=0,
    ).reshape(-1)

    print("Generating risk probabilities (P2)...")

    X_test_tfidf = vectorizer.transform(
        X_test_text
    )

    p2 = risk_model.predict_proba(
        X_test_tfidf
    )[:, 1]

    return p1, p2


def calculate_metrics(
    y_true,
    probabilities,
):
    """
    Calculate classification metrics from probabilities.
    """

    predictions = (
        probabilities >= HYBRID_THRESHOLD
    ).astype(int)

    return {
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(y_true, predictions)
        ),
        "recall": float(
            recall_score(y_true, predictions)
        ),
        "f1_score": float(
            f1_score(y_true, predictions)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
    }


def test_fusion_weights(
    y_test,
    p1,
    p2,
):
    """
    Test multiple P1/P2 weight combinations.
    """

    weight_combinations = [
        (1.0, 0.0),
        (0.9, 0.1),
        (0.8, 0.2),
        (0.7, 0.3),
        (0.6, 0.4),
        (0.5, 0.5),
    ]

    results = []

    print("\nHybrid Fusion Weight Experiment")
    print("-" * 75)

    for alpha, beta in weight_combinations:

        final_probability = (
            alpha * p1
            + beta * p2
        )

        metrics = calculate_metrics(
            y_test,
            final_probability,
        )

        result = {
            "lstm_weight": alpha,
            "risk_weight": beta,
            **metrics,
        }

        results.append(result)

        print(
            f"P1={alpha:.1f} | "
            f"P2={beta:.1f} | "
            f"Accuracy={metrics['accuracy']:.4f} | "
            f"Precision={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f} | "
            f"F1={metrics['f1_score']:.4f} | "
            f"AUC={metrics['roc_auc']:.4f}"
        )

    return results


def find_best_configuration(results):
    """
    Select the configuration with the highest F1 score.
    """

    best = max(
        results,
        key=lambda result: result["f1_score"],
    )

    print("\nBest Hybrid Configuration")
    print("-" * 35)

    print(
        f"LSTM Weight : {best['lstm_weight']:.1f}"
    )

    print(
        f"Risk Weight : {best['risk_weight']:.1f}"
    )

    print(
        f"Accuracy    : {best['accuracy']:.4f}"
    )

    print(
        f"Precision   : {best['precision']:.4f}"
    )

    print(
        f"Recall      : {best['recall']:.4f}"
    )

    print(
        f"F1 Score    : {best['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC     : {best['roc_auc']:.4f}"
    )

    return best


def main():

    create_directory(EVALUATION_DIR)

    (
        dataset,
        X_test_lstm,
        y_test_lstm,
        lstm_model,
        risk_model,
        vectorizer,
    ) = load_resources()

    X_test_text, y_test_text = (
        prepare_text_test_data(dataset)
    )

    # Safety check: both test sets must contain
    # the same labels in the same order.
    if not np.array_equal(
        y_test_lstm,
        y_test_text,
    ):
        raise ValueError(
            "LSTM and text test sets are not aligned. "
            "Hybrid fusion cannot continue safely."
        )

    print(
        f"\nAligned test samples: {len(y_test_lstm)}"
    )

    p1, p2 = generate_probabilities(
        lstm_model,
        risk_model,
        vectorizer,
        X_test_lstm,
        X_test_text,
    )

    results = test_fusion_weights(
        y_test_lstm,
        p1,
        p2,
    )

    best = find_best_configuration(
        results
    )

    experiment = {
        "fusion_formula": (
            "P_final = alpha * P1 + beta * P2"
        ),
        "classification_threshold": HYBRID_THRESHOLD,
        "selection_metric": "f1_score",
        "tested_configurations": results,
        "best_configuration": best,
    }

    save_json(
        experiment,
        os.path.join(
            EVALUATION_DIR,
            "hybrid_fusion_results.json",
        ),
    )

    print(
        "\nHybrid experiment results saved successfully."
    )

    print(
        "\nPhase 7A completed successfully!"
    )


if __name__ == "__main__":
    main()