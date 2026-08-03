import os

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
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

from config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    EVALUATION_DIR,
    SENTIMENT_MODEL_NAME,
    TFIDF_VECTORIZER_NAME,
    TEST_SIZE,
    RANDOM_STATE,
    TFIDF_MAX_FEATURES,
)

from src.utils import (
    create_directory,
    load_pickle,
    save_json,
)


def load_resources():
    """
    Load the sentiment risk dataset,
    Naive Bayes model, and TF-IDF vectorizer.
    """

    print("Loading sentiment risk analyzer resources...")

    dataset_path = os.path.join(
        PROCESSED_DATA_DIR,
        "sentiment_risk_dataset.csv",
    )

    model_path = os.path.join(
        MODELS_DIR,
        SENTIMENT_MODEL_NAME,
    )

    vectorizer_path = os.path.join(
        MODELS_DIR,
        TFIDF_VECTORIZER_NAME,
    )

    df = pd.read_csv(dataset_path)

    model = load_pickle(model_path)

    vectorizer = load_pickle(vectorizer_path)

    print(f"Dataset size: {len(df)}")

    return df, model, vectorizer


def prepare_test_data(df, vectorizer):
    """
    Reproduce the same train-test split used during
    sentiment model training and transform the test text
    using the saved TF-IDF vectorizer.
    """

    X = df["text"]
    y = df["risk_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_test_tfidf = vectorizer.transform(X_test)

    print("\nEvaluation Dataset")
    print("-" * 35)
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    return X_train, X_test_tfidf, y_test


def evaluate_sentiment(
    model,
    X_train,
    X_test_tfidf,
    y_test,
):
    """
    Evaluate the sentiment-aware risk analyzer.
    """

    print("\nRunning risk predictions...")

    predictions = model.predict(
        X_test_tfidf
    )

    # Probability of risk class = 1
    probabilities = model.predict_proba(
        X_test_tfidf
    )[:, 1]

    # ----------------------------------
    # Metrics
    # ----------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
    )

    recall = recall_score(
        y_test,
        predictions,
    )

    f1 = f1_score(
        y_test,
        predictions,
    )

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities,
    )

    roc_auc = auc(
        fpr,
        tpr,
    )

    print("\nSentiment-Aware Risk Analyzer Results")
    print("-" * 45)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    # ----------------------------------
    # Save Experiment Metrics
    # ----------------------------------

    metrics = {
        "model": "TF-IDF + Multinomial Naive Bayes",
        "purpose": "Sentiment-Aware Social Engineering Risk Analysis",
        "label_type": "weak/silver labels",
        "dataset_size": len(X_train) + len(y_test),
        "train_samples": len(X_train),
        "test_samples": len(y_test),
        "tfidf_max_features": TFIDF_MAX_FEATURES,
        "risk_threshold": 2,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
    }

    save_json(
        metrics,
        os.path.join(
            EVALUATION_DIR,
            "sentiment_metrics.json",
        ),
    )

    # ----------------------------------
    # Classification Report
    # ----------------------------------

    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "Lower Risk",
            "Higher Risk",
        ],
    )

    report_path = os.path.join(
        EVALUATION_DIR,
        "sentiment_classification_report.txt",
    )

    with open(
        report_path,
        "w",
    ) as file:
        file.write(report)

    print("\nClassification Report")
    print("-" * 45)
    print(report)

    # ----------------------------------
    # Confusion Matrix
    # ----------------------------------

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Lower Risk",
            "Higher Risk",
        ],
    )

    disp.plot(cmap="Blues")

    plt.title(
        "Sentiment-Aware Risk Analyzer\nConfusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            EVALUATION_DIR,
            "sentiment_confusion_matrix.png",
        )
    )

    plt.close()

    # ----------------------------------
    # ROC Curve
    # ----------------------------------

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

    print(
        "\nSentiment evaluation files saved successfully."
    )


def main():
    """
    Main evaluation pipeline.
    """

    create_directory(EVALUATION_DIR)

    df, model, vectorizer = load_resources()

    (
        X_train,
        X_test_tfidf,
        y_test,
    ) = prepare_test_data(
        df,
        vectorizer,
    )

    evaluate_sentiment(
        model,
        X_train,
        X_test_tfidf,
        y_test,
    )

    print(
        "\nPhase 6C completed successfully!"
    )


if __name__ == "__main__":
    main()