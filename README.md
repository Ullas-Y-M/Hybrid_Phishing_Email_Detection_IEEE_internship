# Hybrid_Phishing_Email_Detection_IEEE_internship
First time training a model 
```
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
