import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

def get_model(cfg: DictConfig):
    """Initialize the model based on the configuration."""

    # Create the model based on the specified architecture
    if cfg.model.name == "FasterRCNN":
        # Instantiate the backbone
        backbone = instantiate(cfg.model.backbone)

        backbone.out_channels = 1280
        
        # Create the Faster R-CNN model
        model = FasterRCNN(
            backbone,
            num_classes=cfg.model.num_classes,
        )
    else:
        raise ValueError(f"Model {cfg.model.name} not supported")

    return model
