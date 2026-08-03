import os

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    SENTIMENT_MODEL_NAME,
    TFIDF_VECTORIZER_NAME,
    TFIDF_MAX_FEATURES,
)

from src.sentiment_preprocessing import calculate_risk_features

from src.utils import (
    create_directory,
    save_pickle,
)


def create_risk_labels(df):
    """
    Generate weak/silver social-engineering risk labels
    for the aligned training emails.
    """

    print("\nGenerating risk labels for training data...")

    risk_features = df["text"].apply(
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

    df = pd.concat(
        [
            df.reset_index(drop=True),
            risk_features.reset_index(drop=True),
        ],
        axis=1,
    )

    df["risk_label"] = (
        df["risk_count"] >= 2
    ).astype(int)

    return df


def load_training_data():
    """
    Load the exact 4200 emails used to train
    the LSTM model.
    """

    training_path = os.path.join(
        PROCESSED_DATA_DIR,
        "train_text.csv",
    )

    print("Loading aligned training dataset...")

    df = pd.read_csv(training_path)

    df = df.dropna(
        subset=["text", "label"]
    ).reset_index(drop=True)

    print(f"Training samples: {len(df)}")

    return df


def create_tfidf(X_train):
    """
    Fit TF-IDF using training emails only.
    """

    print("\nCreating TF-IDF features...")

    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    X_train_tfidf = vectorizer.fit_transform(
        X_train
    )

    print(
        f"TF-IDF training shape: "
        f"{X_train_tfidf.shape}"
    )

    return vectorizer, X_train_tfidf


def train_model(
    X_train_tfidf,
    y_train,
):
    """
    Train Multinomial Naive Bayes using
    weak/silver social-engineering risk labels.
    """

    print(
        "\nTraining Multinomial Naive Bayes "
        "risk analyzer..."
    )

    model = MultinomialNB()

    model.fit(
        X_train_tfidf,
        y_train,
    )

    print("Training completed.")

    return model


def save_models(
    model,
    vectorizer,
):
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

    print(
        f"Naive Bayes model : {model_path}"
    )

    print(
        f"TF-IDF vectorizer : {vectorizer_path}"
    )


def main():
    """
    Train the aligned sentiment-aware
    social-engineering risk analyzer.
    """

    df = load_training_data()

    df = create_risk_labels(df)

    print("\nTraining Risk Label Distribution")
    print("-" * 40)

    print(
        df["risk_label"]
        .value_counts()
        .sort_index()
    )

    X_train = df["text"]

    y_train = df["risk_label"]

    (
        vectorizer,
        X_train_tfidf,
    ) = create_tfidf(
        X_train
    )

    model = train_model(
        X_train_tfidf,
        y_train,
    )

    save_models(
        model,
        vectorizer,
    )

    print(
        "\nAligned sentiment-aware "
        "risk model training completed successfully!"
    )


if __name__ == "__main__":
    main()