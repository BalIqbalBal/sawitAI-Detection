import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

import torch

def get_model(cfg: DictConfig):
    """Initialize the model based on the configuration."""

    # Create the model based on the specified architecture
    if cfg.model.name == "FasterRCNN":
        # Instantiate the backbone
        backbone = instantiate(cfg.model.backbone)
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
        backbone.out_channels = 256  # FPN outputs 256 channels per feature map
       
        # Create the Faster R-CNN model
        model = FasterRCNN(
            backbone,
            num_classes=cfg.model.num_classes,
        )
    else:
        raise ValueError(f"Model {cfg.model.name} not supported")

    return model