import torch
import os


def save_model(model, filename, model_name):
    """
    Save the model to the results folder under a subfolder named after the model.
    
    Args:
        model: The trained model to save.
        filename: The name of the checkpoint file (e.g., "model.pth").
        model_name: The name of the model (e.g., "faster_rcnn").
    """
    # Create the results folder if it doesn't exist
    results_dir = os.path.join("results", model_name)
    os.makedirs(results_dir, exist_ok=True)

    # Save the model checkpoint
    checkpoint_path = os.path.join(results_dir, filename)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")
    
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()