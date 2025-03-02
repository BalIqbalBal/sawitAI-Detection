import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import os
import tifffile
import numpy as np
from PIL import Image

class COCODataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info["file_name"])

        # Load TIFF image
        if image_path.endswith(".tif") or image_path.endswith(".tiff"):
            image = tifffile.imread(image_path)  # Read TIFF image
            image = image.astype(np.float32)  # Convert to float32
            # Normalize if necessary (e.g., scale to [0, 1])
            image = image / np.max(image)
            # Convert to 3 channels if the image is grayscale
            if len(image.shape) == 2:  # Grayscale image
                image = np.stack([image] * 3, axis=-1)
            image = torch.tensor(image).permute(2, 0, 1)  # Convert to CxHxW format
        else:
            # Handle non-TIFF images (e.g., JPEG, PNG)
            image = Image.open(image_path).convert("RGB")
            image = torch.tensor(np.array(image)).float().permute(2, 0, 1)  # Convert to CxHxW format

        # Apply transform if provided
        if self.transform is not None:
            image = self.transform(image)

        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes = []
        labels = []
        for ann in anns:
            boxes.append(ann["bbox"])
            labels.append(ann["category_id"])

        # Convert to tensors
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

        return image, target