def save_model(model, model_path: str):
    """
    Save a trained model to the given path.

    Args:
        model: The trained model to save.
        model_path: Path where the model should be saved.
    """
    import pickle
    import os

    # Create the directory if it doesn't exist
    directory = os.path.dirname(model_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {model_path}")


def load_model(model_path: str):
    """
    Load a trained model from the given path.

    Args:
        model_path: Path to the saved model.

    Returns:
        The loaded model.
    """
    import pickle
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {model_path}")
    return model