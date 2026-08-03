import os
import json

import matplotlib.pyplot as plt
import pandas as pd

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
    SENTIMENT_MODEL_NAME,
    TFIDF_VECTORIZER_NAME,
)

from src.sentiment_preprocessing import calculate_risk_features
from src.utils import load_pickle


def load_resources():
    """
    Load the aligned final test text, trained
    Naive Bayes risk analyzer and TF-IDF vectorizer.
    """

    print(
        "Loading final sentiment-aware "
        "risk analyzer resources..."
    )

    test_path = os.path.join(
        PROCESSED_DATA_DIR,
        "test_text.csv",
    )

    test_df = pd.read_csv(
        test_path
    )

    test_df = test_df.dropna(
        subset=["text"]
    ).reset_index(drop=True)

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
        f"Testing samples: {len(test_df)}"
    )

    return (
        test_df,
        risk_model,
        vectorizer,
    )


def create_test_risk_labels(test_df):
    """
    Generate the same weak/silver risk labels used
    when training the sentiment-aware analyzer.
    """

    print(
        "\nGenerating test risk labels..."
    )

    risk_features = test_df["text"].apply(
        calculate_risk_features
    )

    risk_features.columns = [
        "urgency",
        "fear",
        "financial",
        "credential",
        "action",
        "authority",
        "risk_count",
    ]

    evaluation_df = pd.concat(
        [
            test_df.reset_index(drop=True),
            risk_features.reset_index(drop=True),
        ],
        axis=1,
    )

    evaluation_df["risk_label"] = (
        evaluation_df["risk_count"] >= 2
    ).astype(int)

    print("\nTest Risk Label Distribution")
    print("-" * 40)

    print(
        evaluation_df["risk_label"]
        .value_counts()
        .sort_index()
    )

    return evaluation_df


def evaluate_risk_model(
    evaluation_df,
    risk_model,
    vectorizer,
):
    """
    Evaluate the sentiment-aware Naive Bayes model
    against its social-engineering risk target.
    """

    print(
        "\nCreating TF-IDF features..."
    )

    X_test = vectorizer.transform(
        evaluation_df["text"]
    )

    y_test = evaluation_df[
        "risk_label"
    ].to_numpy()

    print(
        "Running risk predictions..."
    )

    probabilities = risk_model.predict_proba(
        X_test
    )[:, 1]

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
        "model": (
            "Multinomial Naive Bayes "
            "Sentiment-Aware Risk Analyzer"
        ),
        "test_samples": int(
            len(y_test)
        ),
        "target": (
            "Weak/silver social-engineering risk label"
        ),
        "accuracy": float(
            accuracy
        ),
        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
        "f1_score": float(
            f1
        ),
        "roc_auc": float(
            roc_auc
        ),
    }

    print(
        "\nFinal Sentiment-Aware Risk Analyzer Results"
    )

    print("-" * 50)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    return (
        y_test,
        probabilities,
        predictions,
        metrics,
    )


def save_classification_report(
    y_test,
    predictions,
):
    """
    Save risk analyzer classification report.
    """

    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "Lower Risk",
            "Higher Risk",
        ],
    )

    print(
        "\nClassification Report"
    )

    print("-" * 50)

    print(report)

    with open(
        os.path.join(
            EVALUATION_DIR,
            "sentiment_classification_report.txt",
        ),
        "w",
        encoding="utf-8",
    ) as file:

        file.write(report)


def save_confusion_matrix(
    y_test,
    predictions,
):
    """
    Save risk analyzer confusion matrix.
    """

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Lower Risk",
            "Higher Risk",
        ],
    )

    display.plot()

    plt.title(
        "Sentiment-Aware Risk Analyzer "
        "Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            EVALUATION_DIR,
            "sentiment_confusion_matrix.png",
        )
    )

    plt.close()


def save_roc_curve(
    y_test,
    probabilities,
):
    """
    Save risk analyzer ROC curve.
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
        label=(
            f"Risk Analyzer "
            f"(AUC={auc:.4f})"
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
        "Sentiment-Aware Risk Analyzer ROC Curve"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            EVALUATION_DIR,
            "sentiment_roc_curve.png",
        )
    )

    plt.close()


def save_metrics(metrics):
    """
    Save final risk analyzer metrics.
    """

    with open(
        os.path.join(
            EVALUATION_DIR,
            "sentiment_metrics.json",
        ),
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
        test_df,
        risk_model,
        vectorizer,
    ) = load_resources()

    evaluation_df = create_test_risk_labels(
        test_df
    )

    (
        y_test,
        probabilities,
        predictions,
        metrics,
    ) = evaluate_risk_model(
        evaluation_df,
        risk_model,
        vectorizer,
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
        "\nFinal sentiment-aware evaluation "
        "files saved successfully."
    )


if __name__ == "__main__":
    main()