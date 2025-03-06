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
        
        # Convert targets to YOLO format [batch_index, class_id, center_x, center_y, width, height]
        yolo_targets = []
        for batch_idx, target in enumerate(targets):
            boxes = target["boxes"]  # Boxes in xyxy format, shape (N, 4)
            labels = target["labels"]  # Class labels, shape (N,)
            image_height, image_width = images[batch_idx].shape[-2:]  # Get image dimensions

            # Convert to YOLO format
            yolo_boxes = self.convert_xyxy_to_yolo_format(boxes, labels, batch_idx, image_width, image_height)
            yolo_targets.append(yolo_boxes)

        # Concatenate all targets into a single tensor of shape (N_total, 6)
        yolo_targets = torch.cat(yolo_targets, dim=0)

        # Get the features from the backbone
        features = self.model(images)

        if self.training:
            # Compute the loss
            loss_dict = self.loss(features, yolo_targets)

            # Format dictionary to provide clear names for each loss
            return {
                "loss_box": loss_dict[0],  # Box regression loss
                "loss_cls": loss_dict[1],  # Classification loss
                "loss_dfl": loss_dict[2],  # Distribution Focal Loss (DFL)
            }

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
    
    def convert_xyxy_to_yolo_format(
        self, 
        boxes: torch.Tensor, 
        labels: torch.Tensor, 
        batch_index: int, 
        image_width: int, 
        image_height: int
    ) -> torch.Tensor:
        """
        Convert bounding boxes from xyxy format to YOLO format:
        [batch_index, class_id, center_x, center_y, width, height], normalized by image dimensions.

        Args:
            boxes (torch.Tensor): Bounding boxes in xyxy format, shape (N, 4).
            labels (torch.Tensor): Class labels for each box, shape (N,).
            batch_index (int): Index of the current batch.
            image_width (int): Width of the image.
            image_height (int): Height of the image.

        Returns:
            yolo_boxes (torch.Tensor): Bounding boxes in YOLO format, shape (N, 6).
        """
        # Calculate center coordinates, width, and height
        center_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
        center_y = (boxes[:, 1] + boxes[:, 3]) / 2.0
        width = boxes[:, 2] - boxes[:, 0]
        height = boxes[:, 3] - boxes[:, 1]

        # Normalize by image dimensions
        center_x /= image_width
        center_y /= image_height
        width /= image_width
        height /= image_height

        # Add batch index and class labels
        batch_indices = torch.full_like(labels, batch_index, dtype=torch.float32)  # Shape: (N,)
        
        # Combine into YOLO format
        yolo_boxes = torch.stack([
            batch_indices,  # batch_index
            labels.float(),  # class_id
            center_x,  # center_x
            center_y,  # center_y
            width,  # width
            height,  # height
        ], dim=1)  # Shape: (N, 6)

        return yolo_boxes