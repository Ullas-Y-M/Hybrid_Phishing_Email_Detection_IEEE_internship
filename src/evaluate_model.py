import os

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    ConfusionMatrixDisplay,
)

from tensorflow.keras.models import load_model

from config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    MODEL_NAME,
    EVALUATION_DIR,
)

from src.utils import (
    create_directory,
    save_json,
)


def load_data():
    """
    Load the trained LSTM model and test dataset.
    """

    print("Loading model and test dataset...")

    X_test = np.load(
        os.path.join(PROCESSED_DATA_DIR, "X_test.npy")
    )

    y_test = np.load(
        os.path.join(PROCESSED_DATA_DIR, "y_test.npy")
    )

    model = load_model(
        os.path.join(MODELS_DIR, MODEL_NAME)
    )

    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the LSTM model and generate evaluation reports.
    """

    print("\nRunning predictions...")

    probabilities = model.predict(X_test, verbose=0)

    predictions = (probabilities >= 0.5).astype(int)

    # -----------------------------
    # Metrics
    # -----------------------------

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("\nEvaluation Results")
    print("-" * 35)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }

    save_json(
        metrics,
        os.path.join(EVALUATION_DIR, "metrics.json"),
    )

    # -----------------------------
    # Classification Report
    # -----------------------------

    report = classification_report(
        y_test,
        predictions,
        target_names=["Legitimate", "Phishing"],
    )

    report_path = os.path.join(
        EVALUATION_DIR,
        "classification_report.txt",
    )

    with open(report_path, "w") as file:
        file.write(report)

    print("\nClassification Report")
    print("-" * 35)
    print(report)

    # -----------------------------
    # Confusion Matrix
    # -----------------------------

    cm = confusion_matrix(y_test, predictions)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Legitimate", "Phishing"],
    )

    disp.plot(cmap="Blues")

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            EVALUATION_DIR,
            "confusion_matrix.png",
        )
    )

    plt.close()

    # -----------------------------
    # ROC Curve
    # -----------------------------

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {roc_auc:.4f}",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            EVALUATION_DIR,
            "roc_curve.png",
        )
    )

    plt.close()

    print(f"ROC-AUC : {roc_auc:.4f}")

    print("\nEvaluation files saved successfully.")


def main():

    create_directory(EVALUATION_DIR)

    model, X_test, y_test = load_data()

    evaluate_model(
        model,
        X_test,
        y_test,
    )

    print("\nPhase 5 Completed Successfully!")


if __name__ == "__main__":
    main()