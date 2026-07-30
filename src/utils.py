import json
import os
import joblib


def create_directory(directory):
    """
    Create a directory if it does not already exist.
    """
    os.makedirs(directory, exist_ok=True)


def save_json(data, filepath):
    """
    Save a dictionary as a JSON file.
    """
    with open(filepath, "w") as file:
        json.dump(data, file, indent=4)


def load_json(filepath):
    """
    Load data from a JSON file.
    """
    with open(filepath, "r") as file:
        return json.load(file)


def save_pickle(data, filepath):
    """
    Save a Python object using Joblib.
    """
    joblib.dump(data, filepath)


def load_pickle(filepath):
    """
    Load a Python object saved using Joblib.
    """
    return joblib.load(filepath)