import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from omegaconf import DictConfig
from model.yolo.yolo_body import YoloBody
from model.yolo.yolo_training import Loss, ModelEMA
from utils.box_ops import DecodeBox 

class YOLO(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(YOLO, self).__init__()
        # Load model configuration
        self.model_type = cfg.model_type
        self.num_classes = cfg.num_classes
        self.input_shape = cfg.input_shape if "input_shape" in cfg else [640, 640]  # Default input shape
        self.phi = cfg.phi  # Scaling factor for YOLO

        # Initialize the YOLO model
        self.model = YoloBody(cfg)

        # Initialize loss function
        self.loss = Loss(self.model)

        # Initialize EMA if enabled
        self.ema = ModelEMA(self.model) if cfg.get('ema', False) else None

        # Post-processing parameters
        self.letterbox_image = cfg.post_process.letterbox_image if "post_process" in cfg else True
        self.confidence = cfg.post_process.confidence_threshold if "post_process" in cfg else 0.5
        self.nms_iou = cfg.post_process.nms_iou_threshold if "post_process" in cfg else 0.45

        # Initialize DecodeBox for decoding and NMS
        self.bbox_util = DecodeBox(self.num_classes, self.input_shape)

    def forward(
        self, 
        images: List[torch.Tensor], 
        targets: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels`.
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        # Get the features from the backbone
        features = self.model(images)

        if self.training:
            # Compute the loss
            loss_dict = self.loss(features, targets)
            return loss_dict
        else:
            # For inference, return the detections
            detections = self.postprocess(features, images[0].shape[-2:])
            return detections

    def postprocess(self, features: List[torch.Tensor], image_shape: Tuple[int, int]) -> List[Dict[str, torch.Tensor]]:
        """
        Post-process the output from the model to get the final detections.

        Args:
            features (list[Tensor]): features from the model
            image_shape (Tuple[int, int]): Shape of the input image (height, width)

        Returns:
            detections (list[Dict[Tensor]]): the output detections
        """
        # Decode the raw outputs into bounding boxes, scores, and labels
        outputs = self.bbox_util.decode_box(features)

        # Apply Non-Maximum Suppression (NMS)
        results = self.bbox_util.non_max_suppression(
            outputs, 
            self.num_classes, 
            self.input_shape, 
            image_shape, 
            self.letterbox_image, 
            conf_thres=self.confidence, 
            nms_thres=self.nms_iou
        )

        # Format results into a list of dictionaries
        detections = []
        for result in results:
            if result is not None:
                boxes = torch.from_numpy(result[:, :4]).float()  # Convert to tensor
                scores = torch.from_numpy(result[:, 4]).float()  # Confidence scores
                labels = torch.from_numpy(result[:, 5]).long()   # Class labels
                detections.append({
                    'boxes': boxes,
                    'scores': scores,
                    'labels': labels
                })
            else:
                # If no detections, return empty tensors
                detections.append({
                    'boxes': torch.empty((0, 4), dtype=torch.float32),
                    'scores': torch.empty((0,), dtype=torch.float32),
                    'labels': torch.empty((0,), dtype=torch.int64)
                })

        return detections

    def update_ema(self):
        """Update the EMA model if it exists."""
        if self.ema is not None:
            self.ema.update(self.model)

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):
        """Load the state dict into the model."""
        self.model.load_state_dict(state_dict, strict=strict)

    def state_dict(self):
        """Return the state dict of the model."""
        return self.model.state_dict()

    def train(self, mode: bool = True):
        """Set the model to training mode."""
        super(YOLO, self).train(mode)
        self.model.train(mode)

    def eval(self):
        """Set the model to evaluation mode."""
        super(YOLO, self).eval()
        self.model.eval()