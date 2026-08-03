import os
import re
import pandas as pd

from config import PROCESSED_DATA_DIR


# --------------------------------------------------
# Psychological / Social Engineering Keywords
# --------------------------------------------------

URGENCY_WORDS = [
    "urgent",
    "immediately",
    "immediate",
    "now",
    "asap",
    "quickly",
    "today",
    "deadline",
    "expire",
    "expires",
    "limited time",
    "act now",
]

FEAR_WORDS = [
    "suspended",
    "blocked",
    "disabled",
    "terminated",
    "warning",
    "fraud",
    "unauthorized",
    "compromised",
    "security alert",
    "breach",
]

FINANCIAL_WORDS = [
    "money",
    "payment",
    "bank",
    "account",
    "transfer",
    "fund",
    "funds",
    "prize",
    "winner",
    "reward",
    "million",
    "dollar",
    "cash",
    "investment",
]

CREDENTIAL_WORDS = [
    "password",
    "username",
    "login",
    "verify account",
    "verification",
    "credential",
    "pin",
    "otp",
    "confirm identity",
]

ACTION_WORDS = [
    "click",
    "verify",
    "confirm",
    "reply",
    "send",
    "open",
    "download",
    "update",
    "provide",
    "submit",
]

AUTHORITY_WORDS = [
    "administrator",
    "admin",
    "security team",
    "bank manager",
    "director",
    "government",
    "official",
    "department",
    "support team",
]


def contains_keyword(text, keywords):
    """
    Check whether the email contains any keyword
    from a particular psychological-risk category.
    """

    text = str(text).lower()

    for keyword in keywords:
        pattern = r"\b" + re.escape(keyword) + r"\b"

        if re.search(pattern, text):
            return 1

    return 0


def calculate_risk_features(text):
    """
    Extract psychological manipulation indicators
    from an email.
    """

    urgency = contains_keyword(text, URGENCY_WORDS)
    fear = contains_keyword(text, FEAR_WORDS)
    financial = contains_keyword(text, FINANCIAL_WORDS)
    credential = contains_keyword(text, CREDENTIAL_WORDS)
    action = contains_keyword(text, ACTION_WORDS)
    authority = contains_keyword(text, AUTHORITY_WORDS)

    risk_count = (
        urgency
        + fear
        + financial
        + credential
        + action
        + authority
    )

    return pd.Series(
        [
            urgency,
            fear,
            financial,
            credential,
            action,
            authority,
            risk_count,
        ]
    )


def main():

    dataset_path = os.path.join(
        PROCESSED_DATA_DIR,
        "phishing_dataset.csv",
    )

    print("Loading processed dataset...")

    df = pd.read_csv(dataset_path)

    print(f"Dataset size: {len(df)}")

    print("\nExtracting psychological risk features...")

    df[
        [
            "urgency",
            "fear",
            "financial",
            "credential",
            "action",
            "authority",
            "risk_count",
        ]
    ] = df["text"].apply(calculate_risk_features)

    # --------------------------------------------
    # Create Risk Label
    # --------------------------------------------

    # Two or more manipulation indicators
    # indicate psychologically risky language.

    df["risk_label"] = (
        df["risk_count"] >= 2
    ).astype(int)

    print("\nRisk Label Distribution")
    print("-" * 35)

    print(
        df["risk_label"]
        .value_counts()
        .sort_index()
    )

    print("\nRisk Feature Frequency")
    print("-" * 35)

    for feature in [
        "urgency",
        "fear",
        "financial",
        "credential",
        "action",
        "authority",
    ]:

        print(
            f"{feature.capitalize():12}: "
            f"{df[feature].sum()}"
        )

    output_path = os.path.join(
        PROCESSED_DATA_DIR,
        "sentiment_risk_dataset.csv",
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nRisk dataset saved to: {output_path}"
    )

    print(
        "\nPhase 6A completed successfully!"
    )


if __name__ == "__main__":
    main()