import os

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    SENTIMENT_MODEL_NAME,
    TFIDF_VECTORIZER_NAME,
    TFIDF_MAX_FEATURES,
    TEST_SIZE,
    RANDOM_STATE,
)

from src.utils import (
    create_directory,
    save_pickle,
)


def load_dataset():
    """
    Load the sentiment-aware risk dataset.
    """

    dataset_path = os.path.join(
        PROCESSED_DATA_DIR,
        "sentiment_risk_dataset.csv",
    )

    print("Loading sentiment risk dataset...")

    df = pd.read_csv(dataset_path)

    print(f"Dataset size: {len(df)}")

    return df


def split_dataset(df):
    """
    Split the dataset into training and testing sets.
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

    print("\nDataset Split")
    print("-" * 35)
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    return X_train, X_test, y_train, y_test


def create_tfidf(X_train, X_test):
    """
    Convert email text into TF-IDF features.
    """

    print("\nCreating TF-IDF features...")

    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)

    X_test_tfidf = vectorizer.transform(X_test)

    print(
        f"TF-IDF training shape: {X_train_tfidf.shape}"
    )

    print(
        f"TF-IDF testing shape : {X_test_tfidf.shape}"
    )

    return (
        vectorizer,
        X_train_tfidf,
        X_test_tfidf,
    )


def train_model(X_train_tfidf, y_train):
    """
    Train the Multinomial Naive Bayes risk analyzer.
    """

    print("\nTraining Multinomial Naive Bayes model...")

    model = MultinomialNB()

    model.fit(
        X_train_tfidf,
        y_train,
    )

    print("Training completed.")

    return model


def evaluate_model(model, X_test_tfidf, y_test):
    """
    Perform an initial evaluation of the risk analyzer.
    """

    predictions = model.predict(
        X_test_tfidf
    )

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

    print("\nInitial Risk Analyzer Results")
    print("-" * 35)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")


def save_models(model, vectorizer):
    """
    Save the Naive Bayes model and TF-IDF vectorizer.
    """

    create_directory(MODELS_DIR)

    model_path = os.path.join(
        MODELS_DIR,
        SENTIMENT_MODEL_NAME,
    )

    vectorizer_path = os.path.join(
        MODELS_DIR,
        TFIDF_VECTORIZER_NAME,
    )

    save_pickle(
        model,
        model_path,
    )

    save_pickle(
        vectorizer,
        vectorizer_path,
    )

    print("\nModels saved successfully.")
    print(f"Naive Bayes model : {model_path}")
    print(f"TF-IDF vectorizer : {vectorizer_path}")


def main():
    """
    Train the sentiment-aware risk analyzer.
    """

    df = load_dataset()

    X_train, X_test, y_train, y_test = split_dataset(
        df
    )

    (
        vectorizer,
        X_train_tfidf,
        X_test_tfidf,
    ) = create_tfidf(
        X_train,
        X_test,
    )

    model = train_model(
        X_train_tfidf,
        y_train,
    )

    evaluate_model(
        model,
        X_test_tfidf,
        y_test,
    )

    save_models(
        model,
        vectorizer,
    )

    print(
        "\nPhase 6B completed successfully!"
    )


if __name__ == "__main__":
    main()