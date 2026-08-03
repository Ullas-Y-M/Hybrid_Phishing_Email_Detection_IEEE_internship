import os

import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense,
    Dropout,
)
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

from src.utils import (
    create_directory,
    save_pickle,
)


def load_data():
    """
    Load training and validation datasets.
    """

    print("Loading training and validation data...")

    X_train = np.load(
        os.path.join(
            PROCESSED_DATA_DIR,
            "X_train.npy",
        )
    )

    y_train = np.load(
        os.path.join(
            PROCESSED_DATA_DIR,
            "y_train.npy",
        )
    )

    X_val = np.load(
        os.path.join(
            PROCESSED_DATA_DIR,
            "X_val.npy",
        )
    )

    y_val = np.load(
        os.path.join(
            PROCESSED_DATA_DIR,
            "y_val.npy",
        )
    )

    print(f"Training Shape   : {X_train.shape}")
    print(f"Validation Shape : {X_val.shape}")

    return X_train, y_train, X_val, y_val


def build_model():
    """
    Build the LSTM phishing classification model.
    """

    print("\nBuilding LSTM Model...")

    model = Sequential(
        [
            Embedding(
                input_dim=MAX_VOCAB_SIZE,
                output_dim=EMBEDDING_DIM,
                input_length=MAX_SEQUENCE_LENGTH,
            ),

            LSTM(
                LSTM_UNITS
            ),

            Dropout(
                0.5
            ),

            Dense(
                64,
                activation="relu",
            ),

            Dense(
                1,
                activation="sigmoid",
            ),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    return model


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
):
    """
    Train the LSTM using the dedicated validation set.
    """

    print("\nTraining Started...")

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_val,
            y_val,
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            early_stopping
        ],
        verbose=1,
    )

    return history


def save_model_and_history(
    model,
    history,
):
    """
    Save the trained LSTM model and training history.
    """

    create_directory(MODELS_DIR)
    create_directory(EVALUATION_DIR)

    model_path = os.path.join(
        MODELS_DIR,
        MODEL_NAME,
    )

    model.save(
        model_path
    )

    history_path = os.path.join(
        EVALUATION_DIR,
        HISTORY_NAME,
    )

    save_pickle(
        history.history,
        history_path,
    )

    print(
        "\nModel Saved Successfully"
    )

    print(
        f"Model path: {model_path}"
    )


def save_training_graphs(history):
    """
    Save training and validation accuracy/loss graphs.
    """

    create_directory(EVALUATION_DIR)

    # -----------------------------
    # Accuracy Graph
    # -----------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy",
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("LSTM Training and Validation Accuracy")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            EVALUATION_DIR,
            "accuracy.png",
        )
    )

    plt.close()

    # -----------------------------
    # Loss Graph
    # -----------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss",
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LSTM Training and Validation Loss")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            EVALUATION_DIR,
            "loss.png",
        )
    )

    plt.close()

    print(
        "Training graphs saved."
    )


def main():
    """
    Main LSTM training pipeline.
    """

    (
        X_train,
        y_train,
        X_val,
        y_val,
    ) = load_data()

    model = build_model()

    history = train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
    )

    save_model_and_history(
        model,
        history,
    )

    save_training_graphs(
        history
    )

    print(
        "\nLSTM retraining completed successfully!"
    )


if __name__ == "__main__":
    main()