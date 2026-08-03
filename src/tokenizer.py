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
    VALIDATION_SIZE,
    TEST_SIZE,
    RANDOM_STATE,
)


def load_dataset():
    """
    Load the processed phishing email dataset.
    """

    dataset_path = os.path.join(
        PROCESSED_DATA_DIR,
        "phishing_dataset.csv",
    )

    print("Loading processed dataset...")

    df = pd.read_csv(dataset_path)

    # Remove rows with missing text or labels if any exist
    df = df.dropna(
        subset=["text", "label"]
    ).reset_index(drop=True)

    print(f"Dataset size: {len(df)}")

    return df


def split_dataset(df):
    """
    Split the dataset into training, validation,
    and testing sets.

    Split:
        70% Training
        15% Validation
        15% Testing
    """

    texts = df["text"]
    labels = df["label"]

    # --------------------------------------------------
    # First split:
    # 70% training
    # 30% temporary (validation + testing)
    # --------------------------------------------------

    X_train_text, X_temp_text, y_train, y_temp = (
        train_test_split(
            texts,
            labels,
            test_size=(
                VALIDATION_SIZE
                + TEST_SIZE
            ),
            random_state=RANDOM_STATE,
            stratify=labels,
        )
    )

    # --------------------------------------------------
    # Second split:
    # Divide temporary data into validation and test.
    #
    # With validation=0.15 and test=0.15,
    # this becomes a 50/50 split of the temporary set.
    # --------------------------------------------------

    relative_test_size = (
        TEST_SIZE
        / (VALIDATION_SIZE + TEST_SIZE)
    )

    X_val_text, X_test_text, y_val, y_test = (
        train_test_split(
            X_temp_text,
            y_temp,
            test_size=relative_test_size,
            random_state=RANDOM_STATE,
            stratify=y_temp,
        )
    )

    # Reset indices so text and labels remain
    # explicitly aligned when saved.
    X_train_text = X_train_text.reset_index(drop=True)
    X_val_text = X_val_text.reset_index(drop=True)
    X_test_text = X_test_text.reset_index(drop=True)

    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    print("\nDataset Split")
    print("-" * 40)

    print(
        f"Training samples   : {len(X_train_text)}"
    )

    print(
        f"Validation samples : {len(X_val_text)}"
    )

    print(
        f"Testing samples    : {len(X_test_text)}"
    )

    print("\nTraining Class Distribution")
    print("-" * 40)
    print(
        y_train.value_counts().sort_index()
    )

    print("\nValidation Class Distribution")
    print("-" * 40)
    print(
        y_val.value_counts().sort_index()
    )

    print("\nTesting Class Distribution")
    print("-" * 40)
    print(
        y_test.value_counts().sort_index()
    )

    return (
        X_train_text,
        X_val_text,
        X_test_text,
        y_train,
        y_val,
        y_test,
    )


def create_tokenizer(X_train_text):
    """
    Create and fit the tokenizer using ONLY
    the training dataset.

    Validation and test data must not be used
    when learning the vocabulary.
    """

    print("\nCreating tokenizer...")

    tokenizer = Tokenizer(
        num_words=MAX_VOCAB_SIZE,
        oov_token="<OOV>",
    )

    tokenizer.fit_on_texts(
        X_train_text
    )

    print(
        f"Vocabulary learned from "
        f"{len(X_train_text)} training emails."
    )

    print(
        f"Total unique words found: "
        f"{len(tokenizer.word_index)}"
    )

    return tokenizer


def convert_to_sequences(
    tokenizer,
    X_train_text,
    X_val_text,
    X_test_text,
):
    """
    Convert training, validation and testing
    email text into padded integer sequences.
    """

    print("\nConverting text to sequences...")

    train_sequences = tokenizer.texts_to_sequences(
        X_train_text
    )

    val_sequences = tokenizer.texts_to_sequences(
        X_val_text
    )

    test_sequences = tokenizer.texts_to_sequences(
        X_test_text
    )

    X_train = pad_sequences(
        train_sequences,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
    )

    X_val = pad_sequences(
        val_sequences,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
    )

    X_test = pad_sequences(
        test_sequences,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
    )

    print("\nSequence Shapes")
    print("-" * 40)

    print(
        f"Training shape   : {X_train.shape}"
    )

    print(
        f"Validation shape : {X_val.shape}"
    )

    print(
        f"Testing shape    : {X_test.shape}"
    )

    return (
        X_train,
        X_val,
        X_test,
    )


def save_arrays(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
):
    """
    Save tokenized datasets as NumPy arrays.
    """

    print("\nSaving NumPy datasets...")

    np.save(
        os.path.join(
            PROCESSED_DATA_DIR,
            "X_train.npy",
        ),
        X_train,
    )

    np.save(
        os.path.join(
            PROCESSED_DATA_DIR,
            "X_val.npy",
        ),
        X_val,
    )

    np.save(
        os.path.join(
            PROCESSED_DATA_DIR,
            "X_test.npy",
        ),
        X_test,
    )

    np.save(
        os.path.join(
            PROCESSED_DATA_DIR,
            "y_train.npy",
        ),
        y_train.to_numpy(),
    )

    np.save(
        os.path.join(
            PROCESSED_DATA_DIR,
            "y_val.npy",
        ),
        y_val.to_numpy(),
    )

    np.save(
        os.path.join(
            PROCESSED_DATA_DIR,
            "y_test.npy",
        ),
        y_test.to_numpy(),
    )

    print("NumPy datasets saved successfully.")


def save_text_splits(
    X_train_text,
    X_val_text,
    X_test_text,
    y_train,
    y_val,
    y_test,
):
    """
    Save the exact text records used in each split.

    These files allow the Naive Bayes risk analyzer
    and LSTM model to operate on exactly the same
    email records during hybrid fusion.
    """

    print("\nSaving aligned text datasets...")

    train_df = pd.DataFrame(
        {
            "text": X_train_text,
            "label": y_train,
        }
    )

    validation_df = pd.DataFrame(
        {
            "text": X_val_text,
            "label": y_val,
        }
    )

    test_df = pd.DataFrame(
        {
            "text": X_test_text,
            "label": y_test,
        }
    )

    train_df.to_csv(
        os.path.join(
            PROCESSED_DATA_DIR,
            "train_text.csv",
        ),
        index=False,
    )

    validation_df.to_csv(
        os.path.join(
            PROCESSED_DATA_DIR,
            "validation_text.csv",
        ),
        index=False,
    )

    test_df.to_csv(
        os.path.join(
            PROCESSED_DATA_DIR,
            "test_text.csv",
        ),
        index=False,
    )

    print("Aligned text datasets saved successfully.")


def save_tokenizer(tokenizer):
    """
    Save the trained tokenizer.
    """

    os.makedirs(
        MODELS_DIR,
        exist_ok=True,
    )

    tokenizer_path = os.path.join(
        MODELS_DIR,
        "tokenizer.pkl",
    )

    joblib.dump(
        tokenizer,
        tokenizer_path,
    )

    print(
        f"\nTokenizer saved to: {tokenizer_path}"
    )


def main():
    """
    Run the complete tokenization and
    dataset splitting pipeline.
    """

    df = load_dataset()

    (
        X_train_text,
        X_val_text,
        X_test_text,
        y_train,
        y_val,
        y_test,
    ) = split_dataset(df)

    tokenizer = create_tokenizer(
        X_train_text
    )

    (
        X_train,
        X_val,
        X_test,
    ) = convert_to_sequences(
        tokenizer,
        X_train_text,
        X_val_text,
        X_test_text,
    )

    save_arrays(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )

    save_text_splits(
        X_train_text,
        X_val_text,
        X_test_text,
        y_train,
        y_val,
        y_test,
    )

    save_tokenizer(
        tokenizer
    )

    print("\nTokenization Complete")
    print("-" * 40)

    print(
        f"Training data   : {X_train.shape}"
    )

    print(
        f"Validation data : {X_val.shape}"
    )

    print(
        f"Testing data    : {X_test.shape}"
    )

    print(
        "\nPhase 7B data preparation completed successfully!"
    )


if __name__ == "__main__":
    main()