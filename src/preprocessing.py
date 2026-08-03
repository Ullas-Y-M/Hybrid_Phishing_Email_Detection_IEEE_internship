import os
import re

import pandas as pd

from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SAMPLES_PER_CLASS = 3000

ENRON_FILE = "enron.csv"
PHISHING_FILE = "phishing.csv"
OUTPUT_FILE = "phishing_dataset.csv"


# --------------------------------------------------
# Text Cleaning
# --------------------------------------------------

def clean_text(text):
    """
    Clean email text by removing unnecessary content.

    This function is shared between:
    - Dataset preprocessing
    - Model inference

    Keeping one cleaning function prevents differences
    between training-time and prediction-time preprocessing.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Remove HTML tags
    text = re.sub(
        r"<.*?>",
        " ",
        text,
    )

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text,
    )

    # Remove email addresses
    text = re.sub(
        r"\S+@\S+",
        " ",
        text,
    )

    # Remove non-alphabetic characters
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text,
    )

    # Convert to lowercase
    text = text.lower()

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


# --------------------------------------------------
# Load Enron Dataset
# --------------------------------------------------

def load_enron_dataset():
    """
    Load legitimate emails from the Enron dataset.

    The Enron dataset contains:
    Message-ID, Date, From, To, Subject,
    Message, Cc and Bcc.

    Subject and Message are combined to create
    the final email text.
    """

    enron_path = os.path.join(
        RAW_DATA_DIR,
        ENRON_FILE,
    )

    print("Loading Enron dataset...")

    df = pd.read_csv(
        enron_path,
        usecols=[
            "Subject",
            "Message",
        ],
    )

    # Replace missing subject/body values
    df["Subject"] = (
        df["Subject"]
        .fillna("")
        .astype(str)
    )

    df["Message"] = (
        df["Message"]
        .fillna("")
        .astype(str)
    )

    # Combine subject and message
    df["text"] = (
        df["Subject"]
        + " "
        + df["Message"]
    )

    # Keep only required column
    df = df[
        ["text"]
    ].copy()

    # Remove empty emails
    df = df[
        df["text"].str.strip() != ""
    ]

    # Sample legitimate emails
    df = df.sample(
        n=SAMPLES_PER_CLASS,
        random_state=RANDOM_STATE,
    )

    # Legitimate label
    df["label"] = 0

    df = df.reset_index(
        drop=True
    )

    print(
        f"Legitimate emails selected: {len(df)}"
    )

    return df


# --------------------------------------------------
# Load Phishing Dataset
# --------------------------------------------------

def load_phishing_dataset():
    """
    Load phishing emails from the Nigerian Fraud dataset.

    Expected columns include:
    sender, receiver, date, subject,
    body, urls and label.

    Subject and body are combined to create
    the final email text.
    """

    phishing_path = os.path.join(
        RAW_DATA_DIR,
        PHISHING_FILE,
    )

    print("Loading phishing dataset...")

    df = pd.read_csv(
        phishing_path,
        usecols=[
            "subject",
            "body",
        ],
    )

    # Replace missing values
    df["subject"] = (
        df["subject"]
        .fillna("")
        .astype(str)
    )

    df["body"] = (
        df["body"]
        .fillna("")
        .astype(str)
    )

    # Combine subject and body
    df["text"] = (
        df["subject"]
        + " "
        + df["body"]
    )

    df = df[
        ["text"]
    ].copy()

    # Remove empty emails
    df = df[
        df["text"].str.strip() != ""
    ]

    # Make sure enough phishing emails exist
    if len(df) < SAMPLES_PER_CLASS:
        raise ValueError(
            f"Only {len(df)} phishing emails are available. "
            f"{SAMPLES_PER_CLASS} are required."
        )

    # Sample phishing emails
    df = df.sample(
        n=SAMPLES_PER_CLASS,
        random_state=RANDOM_STATE,
    )

    # Phishing label
    df["label"] = 1

    df = df.reset_index(
        drop=True
    )

    print(
        f"Phishing emails selected: {len(df)}"
    )

    return df


# --------------------------------------------------
# Clean Dataset
# --------------------------------------------------

def clean_dataset(df):
    """
    Clean the combined email dataset.

    Cleaning is performed before the final class
    balancing step because some emails may become
    empty after preprocessing.
    """

    print("Cleaning email text...")

    df = df.copy()

    df["text"] = df["text"].apply(
        clean_text
    )

    # Remove empty text after cleaning
    df = df[
        df["text"].str.strip() != ""
    ].copy()

    df = df.reset_index(
        drop=True
    )

    return df


# --------------------------------------------------
# Balance Dataset
# --------------------------------------------------

def balance_dataset(df):
    """
    Ensure that the final processed dataset contains
    an equal number of legitimate and phishing emails.
    """

    legitimate = df[
        df["label"] == 0
    ]

    phishing = df[
        df["label"] == 1
    ]

    minimum_class_size = min(
        len(legitimate),
        len(phishing),
    )

    if minimum_class_size < SAMPLES_PER_CLASS:
        print(
            "\nWarning:"
            "\nSome emails became empty after cleaning."
        )

        print(
            f"Balancing dataset to "
            f"{minimum_class_size} emails per class."
        )

    legitimate = legitimate.sample(
        n=minimum_class_size,
        random_state=RANDOM_STATE,
    )

    phishing = phishing.sample(
        n=minimum_class_size,
        random_state=RANDOM_STATE,
    )

    balanced_df = pd.concat(
        [
            legitimate,
            phishing,
        ],
        ignore_index=True,
    )

    # Shuffle final dataset
    balanced_df = balanced_df.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(
        drop=True
    )

    return balanced_df


# --------------------------------------------------
# Save Dataset
# --------------------------------------------------

def save_dataset(df):
    """
    Save the final processed dataset.
    """

    os.makedirs(
        PROCESSED_DATA_DIR,
        exist_ok=True,
    )

    output_path = os.path.join(
        PROCESSED_DATA_DIR,
        OUTPUT_FILE,
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved to: {output_path}"
    )


# --------------------------------------------------
# Main Preprocessing Pipeline
# --------------------------------------------------

def main():
    """
    Run the complete dataset preprocessing pipeline.

    This function executes only when preprocessing.py
    is run directly using:

        python -m src.preprocessing

    Importing clean_text from another module will NOT
    execute this pipeline.
    """

    # Load legitimate emails
    enron_df = load_enron_dataset()

    # Load phishing emails
    phishing_df = load_phishing_dataset()

    # Merge datasets
    print("Merging datasets...")

    combined_df = pd.concat(
        [
            enron_df,
            phishing_df,
        ],
        ignore_index=True,
    )

    print(
        f"Combined dataset size: {len(combined_df)}"
    )

    # Clean emails
    combined_df = clean_dataset(
        combined_df
    )

    # Ensure balanced final dataset
    final_df = balance_dataset(
        combined_df
    )

    print("\nPreprocessing Complete")
    print("-" * 40)

    print("\nClass Distribution:")

    print(
        final_df["label"]
        .value_counts()
        .sort_index()
    )

    print(
        f"\nFinal dataset size: {len(final_df)}"
    )

    # Save processed dataset
    save_dataset(
        final_df
    )


# --------------------------------------------------
# Entry Point
# --------------------------------------------------

if __name__ == "__main__":
    main()