import re


# --------------------------------------------------
# Social Engineering Indicator Dictionaries
# --------------------------------------------------

RISK_PATTERNS = {
    "urgency": [
        r"\burgent\b",
        r"\bimmediately\b",
        r"\bact now\b",
        r"\bas soon as possible\b",
        r"\basap\b",
        r"\bexpires today\b",
        r"\bwithin 24 hours\b",
        r"\bwithout delay\b",
        r"\btime sensitive\b",
        r"\bfinal warning\b",
    ],

    "fear": [
        r"\bsuspended\b",
        r"\bterminated\b",
        r"\bblocked\b",
        r"\bunauthorized\b",
        r"\bsuspicious activity\b",
        r"\bsecurity alert\b",
        r"\baccount closure\b",
        r"\bpermanently disabled\b",
        r"\blegal action\b",
        r"\bpenalty\b",
    ],

    "financial": [
    r"\bmoney\b",
    r"\bfunds\b",
    r"\bpayment\b",
    r"\bbank\b",
    r"\bbank account\b",
    r"\baccount number\b",
    r"\btransaction\b",
    r"\btransfer\b",
    r"\bwire transfer\b",
    r"\bprize\b",
    r"\breward\b",
    r"\bbeneficiary\b",
    r"\binheritance\b",
    r"\bmillion\b",
    r"\bdollars?\b",
    r"\bcash\b",
    ],

    "credential": [
        r"\bpassword\b",
        r"\busername\b",
        r"\blogin\b",
        r"\blog in\b",
        r"\bsign in\b",
        r"\bcredentials?\b",
        r"\bpin\b",
        r"\bverify your identity\b",
        r"\bconfirm your identity\b",
        r"\baccount information\b",
        r"\bpersonal information\b",
    ],

    "action": [
        r"\bclick\b",
        r"\breply\b",
        r"\brespond\b",
        r"\bverify\b",
        r"\bconfirm\b",
        r"\bsend\b",
        r"\bsubmit\b",
        r"\bdownload\b",
        r"\bopen the attachment\b",
        r"\bcontact us\b",
        r"\bclaim\b",
        r"\bprovide\b",
    ],

    "authority": [
        r"\bsecurity team\b",
        r"\badministrator\b",
        r"\badmin\b",
        r"\bmanagement\b",
        r"\bbank\b",
        r"\bgovernment\b",
        r"\bdepartment\b",
        r"\bofficial\b",
        r"\bmanager\b",
        r"\bdirector\b",
        r"\bceo\b",
        r"\bpolice\b",
        r"\bauthority\b",
    ],
}


# --------------------------------------------------
# Display Names
# --------------------------------------------------

DISPLAY_NAMES = {
    "urgency": "Urgency",
    "fear": "Fear / Threat",
    "financial": "Financial Language",
    "credential": "Credential Request",
    "action": "Action Request",
    "authority": "Authority / Impersonation",
}


# --------------------------------------------------
# Indicator Detection
# --------------------------------------------------

def detect_risk_indicators(text):
    """
    Detect interpretable social-engineering indicators
    in raw email text.

    Returns whether each category was detected and the
    phrases that triggered it.

    This explanation layer does not alter the model's
    prediction.
    """

    if not isinstance(text, str):
        text = str(text)

    normalized_text = text.lower()

    results = {}

    for category, patterns in RISK_PATTERNS.items():

        matches = []

        for pattern in patterns:

            found = re.findall(
                pattern,
                normalized_text,
                flags=re.IGNORECASE,
            )

            for match in found:

                if isinstance(match, tuple):
                    match = " ".join(match)

                match = str(match).strip()

                if (
                    match
                    and match not in matches
                ):
                    matches.append(match)

        results[category] = {
            "detected": len(matches) > 0,
            "matches": matches,
        }

    return results


# --------------------------------------------------
# Explanation Summary
# --------------------------------------------------

def create_explanation(indicators):
    """
    Convert detected indicators into a concise
    human-readable explanation.
    """

    detected_categories = []

    for category, information in indicators.items():

        if information["detected"]:

            detected_categories.append(
                DISPLAY_NAMES[category]
            )

    if not detected_categories:

        return (
            "No major social-engineering indicators "
            "were detected by the rule-based "
            "explanation layer."
        )

    return (
        "Detected social-engineering indicators: "
        + ", ".join(detected_categories)
        + "."
    )


# --------------------------------------------------
# Count Indicators
# --------------------------------------------------

def count_detected_indicators(indicators):
    """
    Count the number of detected social-engineering
    categories.
    """

    return sum(
        1
        for information in indicators.values()
        if information["detected"]
    )


# --------------------------------------------------
# Display Explanation
# --------------------------------------------------

def display_explanation(indicators):
    """
    Display detected social-engineering indicators.
    """

    print("\nDetected Social-Engineering Indicators")
    print("-" * 55)

    detected_any = False

    for category, information in indicators.items():

        name = DISPLAY_NAMES[category]

        if information["detected"]:

            detected_any = True

            print(
                f"{name:<27}: YES"
            )

            if information["matches"]:

                print(
                    " " * 29
                    + "Matched: "
                    + ", ".join(
                        information["matches"]
                    )
                )

        else:

            print(
                f"{name:<27}: NO"
            )

    if not detected_any:

        print(
            "\nNo major social-engineering "
            "indicators detected."
        )