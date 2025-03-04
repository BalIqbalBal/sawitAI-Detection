import torch
import os


def save_model(model, filename):
    """
    Save the model to the results folder under a subfolder named after the model.
    
    Args:
        model: The trained model to save.
        filename: The name of the checkpoint file (e.g., "model.pth").
        model_name: The name of the model (e.g., "faster_rcnn").
    """
    torch.save(model.state_dict(), filename)
    print(f"Model saved to")
    
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()