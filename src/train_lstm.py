import os
from src.utils import create_directory, save_pickle
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    EVALUATION_DIR,
    MAX_VOCAB_SIZE,
    MAX_SEQUENCE_LENGTH,
    EMBEDDING_DIM,
    LSTM_UNITS,
    BATCH_SIZE,
    EPOCHS,
    MODEL_NAME,
    HISTORY_NAME,
)



def load_data():
    print("Loading training data...")

    X_train = np.load(os.path.join(PROCESSED_DATA_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(PROCESSED_DATA_DIR, "X_test.npy"))

    y_train = np.load(os.path.join(PROCESSED_DATA_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(PROCESSED_DATA_DIR, "y_test.npy"))

    print(f"Training Shape : {X_train.shape}")
    print(f"Testing Shape  : {X_test.shape}")

    return X_train, X_test, y_train, y_test


def build_model():

    print("\nBuilding LSTM Model...")

    model = Sequential([
        Embedding(
            input_dim=MAX_VOCAB_SIZE,
            output_dim=EMBEDDING_DIM,
            input_length=MAX_SEQUENCE_LENGTH
        ),

        LSTM(LSTM_UNITS),

        Dropout(0.5),

        Dense(64, activation="relu"),

        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    return model


def train_model(model, X_train, y_train):

    print("\nTraining Started...")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )

    return history


def save_outputs(model, history):

    create_directory(MODELS_DIR)
    create_directory(EVALUATION_DIR)

    save_pickle(
        history.history,
        os.path.join(EVALUATION_DIR, HISTORY_NAME)
   )

    print("\nModel Saved Successfully")


def plot_history(history):

    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    # Accuracy

    plt.figure(figsize=(8,5))
    plt.plot(accuracy, label="Training Accuracy")
    plt.plot(val_accuracy, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(EVALUATION_DIR, "accuracy.png"))
    plt.close()

    # Loss

    plt.figure(figsize=(8,5))
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(EVALUATION_DIR, "loss.png"))
    plt.close()

    print("Training graphs saved.")


def main():

    X_train, X_test, y_train, y_test = load_data()

    model = build_model()

    history = train_model(model, X_train, y_train)

    save_outputs(model, history)

    plot_history(history)

    print("\nPhase 4 Completed Successfully!")


if __name__ == "__main__":
    main()