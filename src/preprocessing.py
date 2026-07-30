"""
preprocessing.py

Preprocesses the Enron and Nigerian Fraud datasets
to create a balanced phishing email dataset.
"""

import re
import pandas as pd
from sklearn.utils import shuffle

# --------------------------------------------------
# Configuration
# --------------------------------------------------

ENRON_PATH = "data/raw/enron.csv"
PHISHING_PATH = "data/raw/phishing.csv"

OUTPUT_PATH = "data/processed/phishing_dataset.csv"

SAMPLE_SIZE = 3000
RANDOM_STATE = 42


# --------------------------------------------------
# Text Cleaning Function
# --------------------------------------------------

def clean_text(text):
    """
    Clean email text by removing unnecessary content.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove non-alphabetic characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Convert to lowercase
    text = text.lower()

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# --------------------------------------------------
# Load and Process Enron Dataset
# --------------------------------------------------

print("Loading Enron dataset...")

enron = pd.read_csv(ENRON_PATH)

# Keep required columns
enron = enron[["Subject", "Message"]]

# Combine subject and message
enron["text"] = (
    enron["Subject"].fillna("")
    + " "
    + enron["Message"].fillna("")
)

# Clean text
enron["text"] = enron["text"].apply(clean_text)

# Remove empty emails
enron = enron[enron["text"].str.len() > 0]

# Randomly sample 3000 emails
enron = enron.sample(
    n=SAMPLE_SIZE,
    random_state=RANDOM_STATE
)

# Assign label
enron["label"] = 0

# Keep only required columns
enron = enron[["text", "label"]]

print(f"Legitimate emails selected: {len(enron)}")


# --------------------------------------------------
# Load and Process Phishing Dataset
# --------------------------------------------------

print("Loading phishing dataset...")

phishing = pd.read_csv(PHISHING_PATH)

# Combine subject and body
phishing["text"] = (
    phishing["subject"].fillna("")
    + " "
    + phishing["body"].fillna("")
)

# Clean text
phishing["text"] = phishing["text"].apply(clean_text)

# Remove empty emails
phishing = phishing[phishing["text"].str.len() > 0]

# Randomly sample 3000 emails
phishing = phishing.sample(
    n=SAMPLE_SIZE,
    random_state=RANDOM_STATE
)

# Assign label
phishing["label"] = 1

# Keep only required columns
phishing = phishing[["text", "label"]]

print(f"Phishing emails selected: {len(phishing)}")


# --------------------------------------------------
# Merge Datasets
# --------------------------------------------------

print("Merging datasets...")

dataset = pd.concat(
    [enron, phishing],
    ignore_index=True
)

print(f"Combined dataset size: {len(dataset)}")


# --------------------------------------------------
# Shuffle Dataset
# --------------------------------------------------

dataset = shuffle(
    dataset,
    random_state=RANDOM_STATE
).reset_index(drop=True)


# --------------------------------------------------
# Save Dataset
# --------------------------------------------------

dataset.to_csv(
    OUTPUT_PATH,
    index=False
)


# --------------------------------------------------
# Display Statistics
# --------------------------------------------------

print("\nPreprocessing Complete")
print("-" * 40)

print("\nClass Distribution:")
print(dataset["label"].value_counts().sort_index())

print(f"\nFinal dataset size: {len(dataset)}")

print(f"\nSaved to: {OUTPUT_PATH}")