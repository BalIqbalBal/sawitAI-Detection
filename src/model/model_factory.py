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

        # Instantiate the anchor generator
        anchor_generator = AnchorGenerator(
            sizes=cfg.model.anchor_generator.sizes,
            aspect_ratios=cfg.model.anchor_generator.aspect_ratios
        )

        # Instantiate the ROI pooler
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=cfg.model.roi_pooler.featmap_names,
            output_size=cfg.model.roi_pooler.output_size,
            sampling_ratio=cfg.model.roi_pooler.sampling_ratio
        )

        # Create the Faster R-CNN model
        model = FasterRCNN(
            backbone,
            num_classes=cfg.model.num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )
    else:
        raise ValueError(f"Model {cfg.model.name} not supported")

    return model
