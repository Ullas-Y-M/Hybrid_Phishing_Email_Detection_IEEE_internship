PROJECT_NAME = "Hybrid Phishing Email Detection"

# -----------------------------
# Paths
# -----------------------------
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
MODELS_DIR = "models"
EVALUATION_DIR = "evaluation"

# -----------------------------
# Dataset
# -----------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2

# -----------------------------
# Tokenizer
# -----------------------------
MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 300

# -----------------------------
# LSTM
# -----------------------------
EMBEDDING_DIM = 128
LSTM_UNITS = 128

# -----------------------------
# Training
# -----------------------------
BATCH_SIZE = 32
EPOCHS = 10
# -----------------------------
# Output Files
# -----------------------------
MODEL_NAME = "lstm_model.keras"
HISTORY_NAME = "history.pkl"
# -----------------------------
# Sentiment / Risk Analyzer
# -----------------------------

SENTIMENT_MODEL_NAME = "sentiment_nb.pkl"
TFIDF_VECTORIZER_NAME = "tfidf_vectorizer.pkl"

TFIDF_MAX_FEATURES = 10000