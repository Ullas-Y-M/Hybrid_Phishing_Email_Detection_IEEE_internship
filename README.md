# Hybrid_Phishing_Email_Detection_IEEE_internship
First time training a model 
```
## 🚀 Live Demo

The Hybrid Phishing Email Detection System is deployed as an interactive Streamlit web application.

### 🔗 Try the Application

**[Launch Hybrid Phishing Email Detector](https://hybrid-phishing-email-detector.streamlit.app/)**

The application allows users to paste email content and receive:

- LSTM phishing probability (P1)
- Sentiment-aware risk probability (P2)
- Final hybrid phishing score
- Phishing / Legitimate classification
- Risk rating
- Social-engineering indicators
- Matched suspicious phrases
- Human-readable risk explanation

> **Note:** The application is a research prototype developed as part of the IEEE Computer Society Bangalore Chapter Student Internship & Mentorship Program 2026. Predictions should not be considered a replacement for production-grade email security systems.
```
## 📊 Final System Performance

The final hybrid framework was evaluated on an untouched test partition of 900 emails.

| Metric | Score |
|---|---:|
| Accuracy | **99.22%** |
| Precision | **99.55%** |
| Recall | **98.89%** |
| F1-Score | **99.22%** |
| ROC-AUC | **99.96%** |

The final hybrid configuration uses:

```text
Final Score = 0.60 × LSTM Probability
            + 0.40 × Social-Engineering Risk Probability
## Project Setup

### Requirements

This project uses Python 3.11 and the following core libraries:

- TensorFlow 2.15.0
- NumPy
- Pandas
- Scikit-learn
- NLTK
- Matplotlib
- BeautifulSoup4
- lxml
- Joblib
- tqdm
```
## Development Environment

This project uses an isolated Python virtual environment (`.venv`) to ensure consistent dependencies and avoid conflicts with other Python projects.

### Create the virtual environment

```bash
python -m venv .venv
```

### Activate it (Windows PowerShell)

```powershell
.\.venv\Scripts\Activate
```

### Verify the interpreter

```bash
python --version
where python
```
## Dependencies

Install all required Python packages using:

```bash
pip install -r requirements.txt
```

### Core Libraries

- TensorFlow 2.15.0
- NumPy 1.26.4
- Pandas 2.2.3
- Scikit-learn 1.5.2
- NLTK 3.9.1
- Matplotlib 3.9.2
- Seaborn 0.13.2
- BeautifulSoup4 4.13.4
- lxml 6.0.1
- Joblib 1.5.2
- tqdm 4.67.1
```
## Project Structure

```
Hybrid_Phishing_Email_Detection_IEEE_internship/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── src/
├── models/
├── notebooks/
├── evaluation/
├── tests/
├── docs/
├── config.py
├── requirements.txt
└── README.md
```

### Directory Overview

- **data/** – Stores raw, processed, and external datasets.
- **src/** – Contains the source code for preprocessing, model training, prediction, and utilities.
- **models/** – Stores trained machine learning models and tokenizers.
- **notebooks/** – Used for experimentation and exploratory data analysis.
- **evaluation/** – Stores evaluation metrics, confusion matrices, and plots.
- **tests/** – Contains unit tests.
- **docs/** – Project documentation and architecture diagrams.
- **config.py** – Central configuration file for model parameters and project settings.
```
## Project Organization

The project follows a modular architecture to separate datasets, source code, trained models, documentation, notebooks, and evaluation artifacts.

### Folder Structure

- `data/raw/` – Original email datasets.
- `data/processed/` – Cleaned and preprocessed datasets.
- `data/external/` – Additional datasets collected during development.
- `src/` – Source code for preprocessing, model training, and prediction.
- `models/` – Saved machine learning models and tokenizers.
- `evaluation/` – Evaluation scripts, metrics, and visualizations.
- `notebooks/` – Jupyter notebooks for experiments.
- `tests/` – Unit tests.
- `docs/` – Documentation and project reports.
```
## Dataset Statistics

| Dataset | Samples |
|----------|---------:|
| Enron (Original) | 517,401 |
| Nigerian Fraud (Original) | 3,332 |

For model training, a balanced dataset will be created by randomly sampling 3,000 legitimate emails from the Enron dataset and 3,000 phishing emails from the Nigerian Fraud dataset, resulting in a final dataset of 6,000 emails.
```
## Data Preprocessing

The preprocessing pipeline performs the following steps:

1. Load the Enron Email Dataset and Nigerian Fraud Dataset.
2. Combine the email subject and body into a single text field.
3. Clean the email text by:
   - Removing HTML tags
   - Removing URLs
   - Removing email addresses
   - Removing special characters
   - Converting text to lowercase
   - Removing extra whitespace
4. Remove empty emails.
5. Randomly sample 3,000 legitimate and 3,000 phishing emails.
6. Merge and shuffle the datasets.
7. Save the processed dataset to:

```
data/processed/phishing_dataset.csv
```

### Final Dataset

| Class | Samples |
|--------|---------:|
| Legitimate | 3000 |
| Phishing | 3000 |
| **Total** | **6000** |
```
## Phase 3 – Tokenization & Data Preparation

The processed dataset is converted into numerical sequences suitable for the LSTM model.

### Steps Performed

- Loaded the processed phishing dataset.
- Split the dataset into:
  - Training Set (80%)
  - Testing Set (20%)
- Created a TensorFlow tokenizer.
- Built a vocabulary of the most frequent words.
- Converted text into integer sequences.
- Applied sequence padding with a maximum length of 300.
- Saved the trained tokenizer.
- Saved the processed NumPy arrays for model training.

### Generated Files
```
## Phase 4 – LSTM Model Training

The phishing email classifier was implemented using an LSTM neural network.

### Model Architecture

- Embedding Layer
- LSTM Layer (128 units)
- Dropout Layer (0.5)
- Dense Layer (64 neurons, ReLU)
- Output Layer (1 neuron, Sigmoid)

### Training Configuration

- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Batch Size: 32
- Epochs: 10
- Early Stopping (patience = 2)

### Outputs

- Trained model (`models/lstm_model.keras`)
- Training history
- Accuracy graph
- Loss graph
```
## Phase 5 – LSTM Model Evaluation

The trained LSTM model was evaluated on the unseen test dataset to measure its performance in phishing email detection.

### Evaluation Process

- Loaded the trained LSTM model.
- Loaded the testing dataset (`X_test.npy` and `y_test.npy`).
- Generated predictions for all test emails.
- Calculated standard machine learning evaluation metrics.
- Generated the confusion matrix and ROC curve.
- Saved the evaluation reports and metrics.

### Evaluation Metrics

| Metric | Value |
|--------|-------:|
| Accuracy | **99.00%** |
| Precision | **99.49%** |
| Recall | **98.50%** |
| F1-Score | **98.99%** |
| ROC-AUC | **99.11%** |

### Generated Evaluation Files

```
evaluation/
├── accuracy.png
├── loss.png
├── confusion_matrix.png
├── roc_curve.png
├── classification_report.txt
├── history.pkl
└── metrics.json
```

### Summary

The LSTM model achieved excellent performance on the balanced phishing email dataset, correctly classifying approximately **99%** of the test emails. The high precision and recall indicate that the model is highly effective at distinguishing phishing emails from legitimate emails while maintaining a low false positive and false negative rate.

The trained LSTM model serves as the primary text classification component of the hybrid phishing email detection framework. In the next phase, a **Sentiment-Aware Risk Analyzer** will be developed using a classical machine learning approach to detect psychological manipulation and social engineering patterns. The outputs of both models will later be combined to improve the overall robustness of the phishing detection system.
```
## Phase 6 – Sentiment-Aware Social Engineering Risk Analyzer

A secondary risk-analysis component was developed to identify psychological manipulation and social-engineering patterns commonly found in phishing emails.

Unlike the LSTM classifier, which directly predicts whether an email is legitimate or phishing, this component estimates the presence of manipulative language and produces a separate risk probability (P₂).

### Psychological Risk Features

The following social-engineering indicators are analyzed:

- Urgency
- Fear and threat language
- Financial incentives or requests
- Credential-related language
- Action-oriented pressure
- Authority-related language

Emails containing two or more manipulation indicators are assigned a higher-risk weak label.

> The risk labels used for this component are automatically generated weak/silver labels based on predefined psychological manipulation indicators rather than manually annotated sentiment labels.

### Risk Dataset Distribution

| Risk Category | Samples |
|---|---:|
| Lower Risk | 2,417 |
| Higher Risk | 3,583 |
| **Total** | **6,000** |

### Risk Feature Frequency

| Feature | Emails |
|---|---:|
| Urgency | 3,230 |
| Fear | 284 |
| Financial | 3,380 |
| Credential | 225 |
| Action | 3,138 |
| Authority | 2,069 |

### Model Architecture

The sentiment-aware risk analyzer uses:

- TF-IDF text vectorization
- Maximum TF-IDF features: 10,000
- Unigram and bigram features
- Multinomial Naive Bayes classifier
- 80:20 train-test split

The model generates a probability representing the estimated social-engineering risk:

P₂ = P(Higher Risk | Email)

### Evaluation Results

| Metric | Value |
|---|---:|
| Accuracy | **84.25%** |
| Precision | **94.90%** |
| Recall | **77.82%** |
| F1-Score | **85.52%** |
| ROC-AUC | **86.36%** |

The high precision indicates that emails classified as higher risk are generally identified reliably according to the generated risk labels. The component is designed to complement the LSTM classifier rather than replace it.

### Generated Models

```text
models/
├── sentiment_nb.pkl
└── tfidf_vectorizer.pkl
```

### Generated Evaluation Files

```text
evaluation/
├── sentiment_metrics.json
├── sentiment_classification_report.txt
├── sentiment_confusion_matrix.png
└── sentiment_roc_curve.png
```

The resulting risk probability P₂ will be combined with the LSTM phishing probability P₁ during the hybrid fusion phase.
```
## Phase 7 – Hybrid Phishing Detection and Probability Fusion

The final stage of the system combines the phishing probability generated by the LSTM classifier with the social-engineering risk probability generated by the sentiment-aware Naive Bayes analyzer.

### Hybrid Architecture

The two model components generate independent probability scores:

- **P₁** – LSTM phishing probability
- **P₂** – Sentiment-aware social-engineering risk probability

The final phishing probability is calculated using weighted probability fusion:

```text
P_final = αP₁ + βP₂
```

where:

```text
α + β = 1
```

### Experimental Dataset Split

To prevent test-set information from influencing model development, the balanced dataset of 6,000 emails was divided into:

| Dataset | Samples | Percentage |
|---|---:|---:|
| Training | 4,200 | 70% |
| Validation | 900 | 15% |
| Testing | 900 | 15% |

Both the LSTM classifier and the sentiment-aware risk analyzer were trained using the same 4,200 training emails.

The validation set was used exclusively for selecting the hybrid fusion weights, while the test set remained untouched until the final evaluation.

### Fusion Weight Selection

Multiple fusion configurations were evaluated using the validation dataset.

| LSTM Weight | Risk Weight | Accuracy | F1-Score | ROC-AUC |
|---:|---:|---:|---:|---:|
| 1.0 | 0.0 | 98.89% | 98.88% | 99.59% |
| 0.9 | 0.1 | 98.89% | 98.88% | 99.89% |
| 0.8 | 0.2 | 98.89% | 98.88% | 99.90% |
| 0.7 | 0.3 | 98.89% | 98.88% | 99.91% |
| 0.6 | 0.4 | 98.89% | 98.88% | 99.91% |
| 0.5 | 0.5 | 98.89% | 98.88% | 99.91% |

Based on validation performance, the selected configuration was:

```text
LSTM Weight (α) = 0.60
Risk Weight (β) = 0.40
```

Therefore, the final hybrid probability is:

```text
P_final = 0.60 × P₁ + 0.40 × P₂
```

The classification threshold is:

```text
P_final >= 0.50 → Phishing
P_final < 0.50  → Legitimate
```

### Final Test Results

The selected fusion weights were frozen before evaluating the final system on the untouched 900-email test dataset.

| Metric | LSTM Only | Hybrid Model |
|---|---:|---:|
| Accuracy | 99.22% | **99.22%** |
| Precision | 99.55% | **99.55%** |
| Recall | 98.89% | **98.89%** |
| F1-Score | 99.22% | **99.22%** |
| ROC-AUC | 99.45% | **99.96%** |

### Final Hybrid Classification Performance

The final test dataset contained:

```text
450 legitimate emails
450 phishing emails
900 total test emails
```

The hybrid model achieved:

- **99.22% accuracy**
- **99.55% precision**
- **98.89% recall**
- **99.22% F1-score**
- **99.96% ROC-AUC**

### Interpretation

The hybrid model preserved the strong classification performance of the LSTM classifier while improving ROC-AUC from **99.45% to 99.96%**.

Although the binary predictions at the 0.50 classification threshold remained unchanged, the sentiment-aware risk probability improved the model's ranking of phishing and legitimate emails.

This demonstrates that psychological and social-engineering risk information can provide a useful complementary signal to sequential language features learned by the LSTM.

### Final Hybrid Architecture

```text
                         Email
                           |
              +------------+------------+
              |                         |
              v                         v
         LSTM Classifier        Sentiment-Aware
                                Risk Analyzer
              |                         |
              v                         v
      P₁ = Phishing Score      P₂ = Risk Score
              |                         |
              +------------+------------+
                           |
                           v
                  Weighted Probability
                         Fusion
                           |
                           v
             P_final = 0.60P₁ + 0.40P₂
                           |
                           v
                    Final Prediction
```

### Generated Evaluation Files

```text
evaluation/
├── fusion_weight_selection.json
├── final_hybrid_metrics.json
├── hybrid_classification_report.txt
├── hybrid_confusion_matrix.png
└── hybrid_roc_comparison.png
```

The final hybrid framework therefore combines sequential language modeling and sentiment-aware social-engineering analysis while maintaining high phishing detection performance.
```
## Phase 8 – End-to-End Phishing Email Prediction

An inference pipeline was implemented to allow the trained hybrid framework to analyze previously unseen email text.

The prediction pipeline uses the same text preprocessing function used during model training to prevent inconsistencies between training-time and inference-time preprocessing.

### Prediction Pipeline

```text
New Email
    |
    v
Text Preprocessing
    |
    +-------------------+
    |                   |
    v                   v
Tokenizer             TF-IDF
    |                   |
    v                   v
LSTM Classifier    Naive Bayes Risk Analyzer
    |                   |
    v                   v
P1                  P2
    |                   |
    +---------+---------+
              |
              v
P_final = 0.60(P1) + 0.40(P2)
              |
              v
      Classification Threshold
              |
       +------+------+
       |             |
       v             v
  Legitimate      Phishing
```

### Running the Predictor

From the project root directory:

```bash
python -m src.predict
```

The user can enter an email directly into the terminal. The system returns:

- LSTM phishing probability (P1)
- Social-engineering risk probability (P2)
- Final hybrid probability
- Binary phishing prediction
- Human-readable risk category

### Example – Phishing Email

```text
LSTM Phishing Probability (P1) : 0.9834
Risk Probability (P2)          : 0.9981
Hybrid Score                   : 0.9893

Prediction                     : PHISHING
Risk Rating                    : VERY HIGH
```

### Example – Legitimate Email

```text
LSTM Phishing Probability (P1) : 0.0039
Risk Probability (P2)          : 0.0306
Hybrid Score                   : 0.0145

Prediction                     : LEGITIMATE
Risk Rating                    : LOW
```

### Generalization Observation

Manual testing showed that the model performs strongly on phishing patterns similar to those represented in the Nigerian Fraud training dataset.

However, a credential-phishing example produced:

```text
P1 = 0.0307
P2 = 0.8259
P_final = 0.3488
Prediction = LEGITIMATE
```

The sentiment-aware component identified substantial social-engineering risk, while the LSTM assigned a low phishing probability.

This indicates a limitation in generalization to phishing styles that differ from the phishing distribution represented in the training dataset.

Future work should therefore evaluate and train the framework using more diverse phishing categories, including credential phishing, account-verification attacks, impersonation, spear phishing, and modern URL-based phishing.

### Risk Rating

For user-facing interpretation, the hybrid probability is mapped to the following display categories:

| Hybrid Score | Display Category |
|---|---|
| < 0.25 | Low |
| 0.25 – < 0.50 | Moderate |
| 0.50 – < 0.75 | High |
| >= 0.75 | Very High |

These categories are intended for presentation and interpretability and are not separately trained or validated risk classes.
```
## Phase 9 – Explainability and Social-Engineering Risk Analysis

An explainability layer was added to provide human-interpretable information about social-engineering cues present in an analyzed email.

The explanation module operates independently of the trained LSTM and Naive Bayes classifiers and does not modify the final prediction.

### Social-Engineering Indicators

The system analyzes emails for six categories of potentially suspicious language:

| Indicator | Description |
|---|---|
| Urgency | Language encouraging immediate action |
| Fear / Threat | Threats, suspension warnings, penalties, or security warnings |
| Financial Language | References to money, transfers, banking, prizes, or financial transactions |
| Credential Request | Requests involving passwords, usernames, login information, or identity verification |
| Action Request | Instructions to click, reply, verify, send, submit, or perform another action |
| Authority / Impersonation | Language referring to organizations, officials, management, banks, or other authority figures |

Pattern matching is performed on the original email text so that interpretable phrases can be displayed to the user.

### Explainability Pipeline

```text
                     New Email
                         |
             +-----------+-----------+
             |                       |
             v                       v
      Hybrid Detection       Explainability Layer
             |                       |
        P1 + P2 Fusion        Pattern Detection
             |                       |
             v                       v
        Prediction          Social-Engineering
                            Risk Indicators
             |                       |
             +-----------+-----------+
                         |
                         v
                 Interpretable Result
```

### Example – Phishing Email

A Nigerian-fraud style email produced:

```text
LSTM Phishing Probability (P1) : 0.9834
Risk Probability (P2)          : 0.9981
Hybrid Score                   : 0.9893

Prediction                     : PHISHING
Risk Rating                    : VERY HIGH
Detected Indicator Categories  : 4
```

The explanation layer detected:

```text
Urgency                    : YES
Financial Language         : YES
Action Request             : YES
Authority / Impersonation  : YES
```

Examples of matched phrases included terms related to immediate action, funds, bank transfers, beneficiaries, and replies.

### Example – Legitimate Email

A normal project meeting email produced:

```text
LSTM Phishing Probability (P1) : 0.0039
Risk Probability (P2)          : 0.0306
Hybrid Score                   : 0.0145

Prediction                     : LEGITIMATE
Risk Rating                    : LOW
Detected Indicator Categories  : 0
```

No major social-engineering indicators were detected.

### Important Interpretation

The explainability module is a rule-based input analysis mechanism.

It identifies observable social-engineering cues in an email but does **not** explain the internal reasoning or hidden representations learned by the LSTM model.

Therefore, the explanation should be interpreted as complementary contextual information rather than a causal explanation of the neural network prediction.
```
## Phase 10 – Streamlit Web Application

A Streamlit-based web interface was developed to provide an interactive demonstration of the hybrid phishing email detection framework.

The application uses the existing inference pipeline without modifying the trained models, fusion weights, or classification threshold.

### Features

The web application allows users to:

- Paste email content for analysis
- Obtain the final phishing or legitimate prediction
- View the LSTM phishing probability (P1)
- View the sentiment-aware risk probability (P2)
- View the final hybrid probability
- View the overall risk rating
- Identify social-engineering indicators
- View matched suspicious phrases
- Receive a human-readable explanation of detected risk indicators

### Running the Application

From the project root directory:

```bash
streamlit run app.py
```

The application loads the trained model resources and launches the phishing detection interface in a web browser.

### Prediction Architecture

```text
                    Email Input
                        |
                        v
                 Text Preprocessing
                        |
              +---------+---------+
              |                   |
              v                   v
         LSTM Model         TF-IDF + NB
              |                   |
              v                   v
             P1                  P2
              |                   |
              +---------+---------+
                        |
                        v
             P_final = 0.60P1 + 0.40P2
                        |
                        v
                Final Prediction
                        |
              +---------+---------+
              |                   |
              v                   v
         Risk Rating       Explainability
                                  |
                                  v
                        Social-Engineering
                           Indicators
```

### Model Loading

Streamlit's resource caching is used to load the trained models once and reuse them across application interactions.

This prevents the LSTM model, tokenizer, Naive Bayes model, and TF-IDF vectorizer from being unnecessarily reloaded whenever the interface refreshes.

### Final Experimental Performance

The application uses the final validated hybrid configuration:

| Metric | Result |
|---|---:|
| Accuracy | 99.22% |
| Precision | 99.55% |
| Recall | 98.89% |
| F1-Score | 99.22% |
| ROC-AUC | 99.96% |

The reported performance corresponds to the held-out 900-email final test partition used during the experimental evaluation.

### Usage Note

The application is intended as a research prototype. Predictions should not be treated as a replacement for comprehensive production email-security systems, particularly for phishing styles that differ substantially from the training distribution.
```
### Cloud Deployment

The final application has been deployed using Streamlit Community Cloud.

**Live Application:**  
[https://hybrid-phishing-email-detector.streamlit.app/](https://hybrid-phishing-email-detector.streamlit.app/)

The deployment uses the same validated hybrid inference pipeline as the local application.

For deployment compatibility, the trained LSTM model is loaded using the HDF5 model artifact while preserving the same trained architecture and weights.

### Deployment Architecture

```text
GitHub Repository
        |
        v
Streamlit Community Cloud
        |
        v
      app.py
        |
        v
+---------------------------+
|     Prediction Pipeline   |
+---------------------------+
        |
   +----+----+
   |         |
   v         v
 LSTM       TF-IDF
 Model        +
  P1       Naive Bayes
              P2
   |         |
   +----+----+
        |
        v
  60/40 Hybrid Fusion
        |
        v
 Final Classification
        |
        +-------------------+
        |                   |
        v                   v
   Risk Rating       Explainability
                            |
                            v
                  Social-Engineering
                      Indicators
```
