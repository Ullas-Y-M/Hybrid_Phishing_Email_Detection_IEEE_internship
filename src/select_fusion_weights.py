import os

import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    EVALUATION_DIR,
    MODEL_NAME,
    SENTIMENT_MODEL_NAME,
    TFIDF_VECTORIZER_NAME,
    HYBRID_THRESHOLD,
)

from src.utils import (
    create_directory,
    load_pickle,
    save_json,
)


# --------------------------------------------------
# Load Models and Validation Data
# --------------------------------------------------

def load_resources():
    """
    Load both trained models and the aligned
    validation dataset.
    """

    print("Loading validation resources...")

    # LSTM validation sequences
    X_val = np.load(
        os.path.join(
            PROCESSED_DATA_DIR,
            "X_val.npy",
        )
    )

    y_val = np.load(
        os.path.join(
            PROCESSED_DATA_DIR,
            "y_val.npy",
        )
    )

    # Exact validation email text
    validation_df = pd.read_csv(
        os.path.join(
            PROCESSED_DATA_DIR,
            "validation_text.csv",
        )
    )

    # LSTM model
    lstm_model = load_model(
        os.path.join(
            MODELS_DIR,
            MODEL_NAME,
        )
    )

    # Naive Bayes model
    risk_model = load_pickle(
        os.path.join(
            MODELS_DIR,
            SENTIMENT_MODEL_NAME,
        )
    )

    # TF-IDF vectorizer
    vectorizer = load_pickle(
        os.path.join(
            MODELS_DIR,
            TFIDF_VECTORIZER_NAME,
        )
    )

    print(
        f"Validation samples: {len(y_val)}"
    )

    return (
        X_val,
        y_val,
        validation_df,
        lstm_model,
        risk_model,
        vectorizer,
    )


# --------------------------------------------------
# Verify Alignment
# --------------------------------------------------

def verify_alignment(
    y_val,
    validation_df,
):
    """
    Verify that the LSTM validation sequences
    and validation text represent the same
    records in the same order.
    """

    text_labels = (
        validation_df["label"]
        .to_numpy()
    )

    if len(y_val) != len(text_labels):
        raise ValueError(
            "Validation dataset sizes do not match."
        )

    if not np.array_equal(
        y_val,
        text_labels,
    ):
        raise ValueError(
            "Validation labels are not aligned."
        )

    print(
        "Validation alignment verified."
    )


# --------------------------------------------------
# Generate P1 and P2
# --------------------------------------------------

def generate_probabilities(
    X_val,
    validation_df,
    lstm_model,
    risk_model,
    vectorizer,
):
    """
    Generate validation probabilities from
    both model components.

    P1 = LSTM phishing probability
    P2 = Naive Bayes social-engineering risk probability
    """

    print(
        "\nGenerating LSTM probabilities (P1)..."
    )

    p1 = lstm_model.predict(
        X_val,
        verbose=0,
    ).reshape(-1)

    print(
        "Generating risk probabilities (P2)..."
    )

    X_val_tfidf = vectorizer.transform(
        validation_df["text"]
    )

    p2 = risk_model.predict_proba(
        X_val_tfidf
    )[:, 1]

    print(
        "Probability generation completed."
    )

    return p1, p2


# --------------------------------------------------
# Calculate Metrics
# --------------------------------------------------

def calculate_metrics(
    y_true,
    probabilities,
):
    """
    Calculate evaluation metrics for a
    probability vector.
    """

    predictions = (
        probabilities
        >= HYBRID_THRESHOLD
    ).astype(int)

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),

        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "f1_score": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
    }


# --------------------------------------------------
# Fusion Weight Experiment
# --------------------------------------------------

def test_weights(
    y_val,
    p1,
    p2,
):
    """
    Test candidate fusion weights using ONLY
    the validation dataset.
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

    print(
        "\nValidation Fusion Weight Experiment"
    )

    print("-" * 90)

    for alpha, beta in weight_combinations:

        final_probability = (
            alpha * p1
            + beta * p2
        )

        metrics = calculate_metrics(
            y_val,
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


# --------------------------------------------------
# Select Best Configuration
# --------------------------------------------------

def select_best_configuration(results):
    """
    Select fusion weights.

    Primary criterion:
        F1 score

    Tie-breakers:
        ROC-AUC
        Accuracy
    """

    best = max(
        results,
        key=lambda result: (
            result["f1_score"],
            result["roc_auc"],
            result["accuracy"],
        ),
    )

    print(
        "\nSelected Hybrid Configuration"
    )

    print("-" * 40)

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


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    create_directory(
        EVALUATION_DIR
    )

    (
        X_val,
        y_val,
        validation_df,
        lstm_model,
        risk_model,
        vectorizer,
    ) = load_resources()

    verify_alignment(
        y_val,
        validation_df,
    )

    p1, p2 = generate_probabilities(
        X_val,
        validation_df,
        lstm_model,
        risk_model,
        vectorizer,
    )

    results = test_weights(
        y_val,
        p1,
        p2,
    )

    best = select_best_configuration(
        results
    )

    experiment = {
        "experiment": (
            "Validation-Based Hybrid Fusion "
            "Weight Selection"
        ),

        "validation_samples": int(
            len(y_val)
        ),

        "fusion_formula": (
            "P_final = alpha * P1 + beta * P2"
        ),

        "p1": (
            "LSTM phishing probability"
        ),

        "p2": (
            "Naive Bayes social-engineering "
            "risk probability"
        ),

        "classification_threshold": (
            HYBRID_THRESHOLD
        ),

        "selection_metric": (
            "F1 score with ROC-AUC and "
            "accuracy tie-breakers"
        ),

        "tested_configurations": results,

        "selected_configuration": best,
    }

    output_path = os.path.join(
        EVALUATION_DIR,
        "fusion_weight_selection.json",
    )

    save_json(
        experiment,
        output_path,
    )

    print(
        "\nValidation experiment saved to:"
    )

    print(output_path)

    print(
        "\nPhase 7B completed successfully!"
    )


if __name__ == "__main__":
    main()