import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

import torch

def build_faster_rcnn(cfg):
    """Build the Faster R-CNN model based on the configuration."""
    backbone = resnet_fpn_backbone(
        backbone_name=cfg.backbone.backbone_name,
        pretrained=cfg.backbone.pretrained,
    )

    model = FasterRCNN(
        backbone,
        num_classes=cfg.num_classes,
        )
    
    return model

def build_yolo(cfg):
    """Placeholder for YOLO model initialization."""
    pass

def build_detr(cfg):
    """Placeholder for DETR model initialization."""
    pass

def get_model(cfg: DictConfig):
    """Return the model instance based on the specified architecture."""
    if cfg.model.model_type == "FasterRCNN":
        return build_faster_rcnn(cfg.model)
    elif cfg.model.model_type == "YOLO":
        return build_yolo(cfg.model)
    elif cfg.model.model_type == "DETR":
        return build_detr(cfg.model)
    else:
        raise ValueError(f"Model {cfg.model.name} not supported")
