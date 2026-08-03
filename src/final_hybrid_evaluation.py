import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    ConfusionMatrixDisplay,
)

from config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    EVALUATION_DIR,
    MODEL_NAME,
    SENTIMENT_MODEL_NAME,
    TFIDF_VECTORIZER_NAME,
    HYBRID_THRESHOLD,
    LSTM_WEIGHT,
    RISK_WEIGHT,
)

from src.utils import (
    create_directory,
    load_pickle,
    save_json,
)


def load_resources():
    """
    Load the frozen models and untouched test dataset.
    """

    print("Loading final test resources...")

    X_test = np.load(
        os.path.join(
            PROCESSED_DATA_DIR,
            "X_test.npy",
        )
    )

    y_test = np.load(
        os.path.join(
            PROCESSED_DATA_DIR,
            "y_test.npy",
        )
    )

    test_df = pd.read_csv(
        os.path.join(
            PROCESSED_DATA_DIR,
            "test_text.csv",
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

    print(
        f"Final test samples: {len(y_test)}"
    )

    return (
        X_test,
        y_test,
        test_df,
        lstm_model,
        risk_model,
        vectorizer,
    )


def verify_alignment(
    y_test,
    test_df,
):
    """
    Verify that sequence and text test datasets
    correspond to the same records.
    """

    text_labels = (
        test_df["label"]
        .to_numpy()
    )

    if len(y_test) != len(text_labels):
        raise ValueError(
            "Test dataset sizes do not match."
        )

    if not np.array_equal(
        y_test,
        text_labels,
    ):
        raise ValueError(
            "Test datasets are not aligned."
        )

    print(
        "Test alignment verified."
    )


def generate_probabilities(
    X_test,
    test_df,
    lstm_model,
    risk_model,
    vectorizer,
):
    """
    Generate P1 and P2 for the untouched test set.
    """

    print(
        "\nGenerating LSTM probabilities (P1)..."
    )

    p1 = lstm_model.predict(
        X_test,
        verbose=0,
    ).reshape(-1)

    print(
        "Generating risk probabilities (P2)..."
    )

    X_test_tfidf = vectorizer.transform(
        test_df["text"]
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
    Calculate binary classification metrics.
    """

    predictions = (
        probabilities >= HYBRID_THRESHOLD
    ).astype(int)

    metrics = {
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

    return predictions, metrics


def print_metrics(
    name,
    metrics,
):
    """
    Display evaluation metrics.
    """

    print(f"\n{name}")
    print("-" * 45)

    print(
        f"Accuracy : {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {metrics['precision']:.4f}"
    )

    print(
        f"Recall   : {metrics['recall']:.4f}"
    )

    print(
        f"F1 Score : {metrics['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC  : {metrics['roc_auc']:.4f}"
    )


def save_confusion_matrix(
    y_test,
    predictions,
):
    """
    Save final hybrid confusion matrix.
    """

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Legitimate",
            "Phishing",
        ],
    )

    disp.plot(
        cmap="Blues"
    )

    plt.title(
        "Final Hybrid Model Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            EVALUATION_DIR,
            "hybrid_confusion_matrix.png",
        )
    )

    plt.close()


def save_roc_comparison(
    y_test,
    p1,
    hybrid_probability,
):
    """
    Compare LSTM-only and Hybrid ROC curves.
    """

    lstm_fpr, lstm_tpr, _ = roc_curve(
        y_test,
        p1,
    )

    hybrid_fpr, hybrid_tpr, _ = roc_curve(
        y_test,
        hybrid_probability,
    )

    lstm_auc = roc_auc_score(
        y_test,
        p1,
    )

    hybrid_auc = roc_auc_score(
        y_test,
        hybrid_probability,
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        lstm_fpr,
        lstm_tpr,
        label=(
            f"LSTM "
            f"(AUC={lstm_auc:.4f})"
        ),
    )

    plt.plot(
        hybrid_fpr,
        hybrid_tpr,
        label=(
            f"Hybrid "
            f"(AUC={hybrid_auc:.4f})"
        ),
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "LSTM vs Hybrid ROC Curve"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            EVALUATION_DIR,
            "hybrid_roc_comparison.png",
        )
    )

    plt.close()


def main():

    create_directory(
        EVALUATION_DIR
    )

    (
        X_test,
        y_test,
        test_df,
        lstm_model,
        risk_model,
        vectorizer,
    ) = load_resources()

    verify_alignment(
        y_test,
        test_df,
    )

    p1, p2 = generate_probabilities(
        X_test,
        test_df,
        lstm_model,
        risk_model,
        vectorizer,
    )

    # ----------------------------------
    # LSTM-only baseline
    # ----------------------------------

    (
        lstm_predictions,
        lstm_metrics,
    ) = calculate_metrics(
        y_test,
        p1,
    )

    # ----------------------------------
    # Frozen Hybrid Model
    # ----------------------------------

    hybrid_probability = (
        LSTM_WEIGHT * p1
        + RISK_WEIGHT * p2
    )

    (
        hybrid_predictions,
        hybrid_metrics,
    ) = calculate_metrics(
        y_test,
        hybrid_probability,
    )

    print_metrics(
        "LSTM-Only Test Results",
        lstm_metrics,
    )

    print_metrics(
        "Final Hybrid Test Results",
        hybrid_metrics,
    )

    # ----------------------------------
    # Classification Report
    # ----------------------------------

    report = classification_report(
        y_test,
        hybrid_predictions,
        target_names=[
            "Legitimate",
            "Phishing",
        ],
    )

    print(
        "\nFinal Hybrid Classification Report"
    )

    print("-" * 45)

    print(report)

    with open(
        os.path.join(
            EVALUATION_DIR,
            "hybrid_classification_report.txt",
        ),
        "w",
    ) as file:

        file.write(report)

    # ----------------------------------
    # Save figures
    # ----------------------------------

    save_confusion_matrix(
        y_test,
        hybrid_predictions,
    )

    save_roc_comparison(
        y_test,
        p1,
        hybrid_probability,
    )

    # ----------------------------------
    # Save experiment
    # ----------------------------------

    experiment = {
        "experiment": (
            "Final Hybrid Phishing "
            "Detection Evaluation"
        ),

        "test_samples": int(
            len(y_test)
        ),

        "fusion_formula": (
            "P_final = 0.60 * P1 + 0.40 * P2"
        ),

        "lstm_weight": LSTM_WEIGHT,

        "risk_weight": RISK_WEIGHT,

        "classification_threshold": (
            HYBRID_THRESHOLD
        ),

        "weight_selection": (
            "Weights selected using validation "
            "dataset only"
        ),

        "lstm_baseline": lstm_metrics,

        "hybrid_model": hybrid_metrics,
    }

    save_json(
        experiment,
        os.path.join(
            EVALUATION_DIR,
            "final_hybrid_metrics.json",
        ),
    )

    print(
        "\nFinal hybrid evaluation files "
        "saved successfully."
    )

    print(
        "\nPhase 7C completed successfully!"
    )


if __name__ == "__main__":
    main()