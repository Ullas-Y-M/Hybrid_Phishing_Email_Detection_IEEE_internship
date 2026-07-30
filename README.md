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
