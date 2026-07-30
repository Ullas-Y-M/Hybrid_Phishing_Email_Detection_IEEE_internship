import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    MAX_VOCAB_SIZE,
    MAX_SEQUENCE_LENGTH,
    TEST_SIZE,
    RANDOM_STATE,
)


def load_dataset():
    """Load the processed phishing dataset."""
    dataset_path = os.path.join(PROCESSED_DATA_DIR, "phishing_dataset.csv")

    print("Loading processed dataset...")
    df = pd.read_csv(dataset_path)

    return df


def split_dataset(df):
    """Split dataset into train and test sets."""

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nDataset Split")
    print("-------------------------")
    print(f"Training samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")

    return X_train, X_test, y_train, y_test


def create_tokenizer(X_train):
    """Create and fit tokenizer."""

    tokenizer = Tokenizer(
        num_words=MAX_VOCAB_SIZE,
        oov_token="<OOV>"
    )

    tokenizer.fit_on_texts(X_train)

    print("\nTokenizer Created")
    print("-------------------------")
    print(f"Vocabulary Size: {len(tokenizer.word_index)}")

    return tokenizer


def convert_sequences(tokenizer, X_train, X_test):
    """Convert text into padded sequences."""

    train_sequences = tokenizer.texts_to_sequences(X_train)
    test_sequences = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(
        train_sequences,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
    )

    X_test_pad = pad_sequences(
        test_sequences,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
    )

    print("\nSequence Conversion Complete")
    print("-------------------------")
    print(f"Training Shape : {X_train_pad.shape}")
    print(f"Testing Shape  : {X_test_pad.shape}")

    return X_train_pad, X_test_pad


def save_files(tokenizer, X_train, X_test, y_train, y_test):
    """Save tokenizer and processed arrays."""

    os.makedirs(MODELS_DIR, exist_ok=True)

    tokenizer_path = os.path.join(MODELS_DIR, "tokenizer.pkl")
    joblib.dump(tokenizer, tokenizer_path)

    np.save(os.path.join(PROCESSED_DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, "X_test.npy"), X_test)

    np.save(os.path.join(PROCESSED_DATA_DIR, "y_train.npy"), y_train)

    np.save(os.path.join(PROCESSED_DATA_DIR, "y_test.npy"), y_test)

    print("\nFiles Saved Successfully")
    print("-------------------------")
    print(f"Tokenizer : {tokenizer_path}")
    print(f"Arrays     : {PROCESSED_DATA_DIR}")


def main():

    df = load_dataset()

    X_train, X_test, y_train, y_test = split_dataset(df)

    tokenizer = create_tokenizer(X_train)

    X_train_pad, X_test_pad = convert_sequences(
        tokenizer,
        X_train,
        X_test,
    )

    save_files(
        tokenizer,
        X_train_pad,
        X_test_pad,
        y_train,
        y_test,
    )

    print("\nPhase 3 Completed Successfully!")


if __name__ == "__main__":
    main()