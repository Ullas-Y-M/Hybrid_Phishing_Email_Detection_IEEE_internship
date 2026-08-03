import os
import json

import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)

from config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    EVALUATION_DIR,
    MODEL_NAME,
)


def load_resources():
    """
    Load the final trained LSTM model and
    the untouched aligned test dataset.
    """

    print("Loading final LSTM model and test dataset...")

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

    model = load_model(
        os.path.join(
            MODELS_DIR,
            MODEL_NAME,
        )
    )

    print(f"Testing samples: {len(y_test)}")
    print(f"Testing shape  : {X_test.shape}")

    return model, X_test, y_test


def evaluate_model(
    model,
    X_test,
    y_test,
):
    """
    Evaluate the LSTM classifier using the
    untouched final test dataset.
    """

    print("\nRunning LSTM predictions...")

    probabilities = model.predict(
        X_test,
        verbose=0,
    ).reshape(-1)

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    metrics = {
        "model": "LSTM",
        "test_samples": int(len(y_test)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
    }

    print("\nFinal LSTM Test Results")
    print("-" * 40)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    return (
        probabilities,
        predictions,
        metrics,
    )


def save_classification_report(
    y_test,
    predictions,
):
    """
    Generate and save the classification report.
    """

    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "Legitimate",
            "Phishing",
        ],
    )

    print("\nClassification Report")
    print("-" * 40)
    print(report)

    report_path = os.path.join(
        EVALUATION_DIR,
        "classification_report.txt",
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report)


def save_confusion_matrix(
    y_test,
    predictions,
):
    """
    Generate and save the LSTM confusion matrix.
    """

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Legitimate",
            "Phishing",
        ],
    )

    display.plot()

    plt.title(
        "LSTM Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            EVALUATION_DIR,
            "confusion_matrix.png",
        )
    )

    plt.close()


def save_roc_curve(
    y_test,
    probabilities,
):
    """
    Generate and save the LSTM ROC curve.
    """

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities,
    )

    auc = roc_auc_score(
        y_test,
        probabilities,
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        fpr,
        tpr,
        label=f"LSTM (AUC={auc:.4f})",
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
        "LSTM ROC Curve"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            EVALUATION_DIR,
            "roc_curve.png",
        )
    )

    plt.close()


def save_metrics(metrics):
    """
    Save final LSTM evaluation metrics.
    """

    metrics_path = os.path.join(
        EVALUATION_DIR,
        "metrics.json",
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )


def main():

    os.makedirs(
        EVALUATION_DIR,
        exist_ok=True,
    )

    (
        model,
        X_test,
        y_test,
    ) = load_resources()

    (
        probabilities,
        predictions,
        metrics,
    ) = evaluate_model(
        model,
        X_test,
        y_test,
    )

    save_classification_report(
        y_test,
        predictions,
    )

    save_confusion_matrix(
        y_test,
        predictions,
    )

    save_roc_curve(
        y_test,
        probabilities,
    )

    save_metrics(
        metrics
    )

    print(
        "\nFinal LSTM evaluation files saved successfully."
    )


if __name__ == "__main__":
    main()