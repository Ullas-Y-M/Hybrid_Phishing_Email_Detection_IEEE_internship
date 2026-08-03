import streamlit as st

from src.predict import (
    load_models,
    predict_email,
)

from src.explainability import DISPLAY_NAMES


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Hybrid Phishing Email Detector",
    page_icon="🛡️",
    layout="wide",
)


# --------------------------------------------------
# Cache Models
# --------------------------------------------------

@st.cache_resource
def load_cached_models():
    """
    Load ML models once and cache them so that
    Streamlit does not reload the models after
    every user interaction.
    """

    return load_models()


# --------------------------------------------------
# Load Models
# --------------------------------------------------

try:

    (
        lstm_model,
        tokenizer,
        risk_model,
        vectorizer,
    ) = load_cached_models()

except Exception as error:

    st.error(
        f"Unable to load model resources: {error}"
    )

    st.stop()


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title(
    "🛡️ Hybrid Phishing Email Detector"
)

st.write(
    """
    Analyze email text using a hybrid phishing detection
    framework combining an LSTM classifier with a
    sentiment-aware social-engineering risk analyzer.
    """
)

st.divider()


# --------------------------------------------------
# Email Input
# --------------------------------------------------

st.subheader(
    "Email Analysis"
)

email_text = st.text_area(
    "Paste the email content below:",
    height=250,
    placeholder=(
        "Example:\n\n"
        "URGENT SECURITY ALERT\n\n"
        "Your account has been suspended..."
    ),
)


# --------------------------------------------------
# Analyze Button
# --------------------------------------------------

analyze_button = st.button(
    "Analyze Email",
    type="primary",
    use_container_width=True,
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if analyze_button:

    if not email_text.strip():

        st.warning(
            "Please enter email content before analysis."
        )

    else:

        try:

            with st.spinner(
                "Analyzing email..."
            ):

                result = predict_email(
                    email_text,
                    lstm_model,
                    tokenizer,
                    risk_model,
                    vectorizer,
                )

            # --------------------------------------
            # Final Prediction
            # --------------------------------------

            st.divider()

            st.subheader(
                "Detection Result"
            )

            prediction = result[
                "prediction"
            ]

            risk_rating = result[
                "risk_rating"
            ]

            if prediction == "PHISHING":

                st.error(
                    "⚠️ PHISHING EMAIL DETECTED"
                )

            else:

                st.success(
                    "✅ EMAIL CLASSIFIED AS LEGITIMATE"
                )

            st.write(
                f"**Risk Rating:** {risk_rating}"
            )

            # --------------------------------------
            # Probability Metrics
            # --------------------------------------

            st.subheader(
                "Model Scores"
            )

            col1, col2, col3 = st.columns(
                3
            )

            with col1:

                st.metric(
                    "LSTM Probability (P₁)",
                    (
                        f"{result['lstm_probability'] * 100:.2f}%"
                    ),
                )

            with col2:

                st.metric(
                    "Risk Probability (P₂)",
                    (
                        f"{result['risk_probability'] * 100:.2f}%"
                    ),
                )

            with col3:

                st.metric(
                    "Hybrid Score",
                    (
                        f"{result['hybrid_score'] * 100:.2f}%"
                    ),
                )

            st.caption(
                "Final score = 60% LSTM probability "
                "+ 40% social-engineering risk probability."
            )

            # --------------------------------------
            # Hybrid Probability Progress
            # --------------------------------------

            st.subheader(
                "Hybrid Risk Score"
            )

            st.progress(
                min(
                    max(
                        float(
                            result["hybrid_score"]
                        ),
                        0.0,
                    ),
                    1.0,
                )
            )

            st.write(
                f"{result['hybrid_score'] * 100:.2f}%"
            )

            # --------------------------------------
            # Explainability
            # --------------------------------------

            st.divider()

            st.subheader(
                "Social-Engineering Indicators"
            )

            st.write(
                "Detected indicator categories: "
                f"**{result['indicator_count']}**"
            )

            indicators = result[
                "indicators"
            ]

            for (
                category,
                information,
            ) in indicators.items():

                display_name = DISPLAY_NAMES[
                    category
                ]

                if information[
                    "detected"
                ]:

                    st.warning(
                        f"⚠️ {display_name}"
                    )

                    matches = information[
                        "matches"
                    ]

                    if matches:

                        st.write(
                            "**Matched phrases:** "
                            + ", ".join(matches)
                        )

                else:

                    st.write(
                        f"✓ {display_name}: "
                        "Not detected"
                    )

            # --------------------------------------
            # Explanation
            # --------------------------------------

            st.subheader(
                "Explanation"
            )

            st.info(
                result["explanation"]
            )

            st.caption(
                "The social-engineering indicators are "
                "rule-based contextual explanations. "
                "They do not represent the internal "
                "reasoning of the LSTM model."
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )


# --------------------------------------------------
# System Information
# --------------------------------------------------

st.divider()

with st.expander(
    "About the Detection Framework"
):

    st.markdown(
        """
        ### Hybrid Architecture

        The detector combines two probability signals:

        **P₁ — LSTM phishing classifier**

        Learns sequential language patterns associated
        with legitimate and phishing emails.

        **P₂ — Sentiment-aware risk analyzer**

        Estimates social-engineering risk using a
        TF-IDF representation and Multinomial Naive Bayes.

        The final probability is calculated as:

        ```
        P_final = 0.60(P₁) + 0.40(P₂)
        ```

        Emails with a final score of at least **0.50**
        are classified as phishing.

        ### Final Test Performance

        - Accuracy: **99.22%**
        - Precision: **99.55%**
        - Recall: **98.89%**
        - F1-score: **99.22%**
        - ROC-AUC: **99.96%**

        These results were obtained on the held-out
        900-email final test partition of the experimental
        dataset.
        """
    )