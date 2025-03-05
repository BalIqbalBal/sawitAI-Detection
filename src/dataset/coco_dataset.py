import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import os

class COCODataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        """
        Args:
            root_dir (str): Path to the directory with images.
            annotation_file (str): Path to the COCO annotation file (e.g., `instances_train2017.json`).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.transform = transform

        # Preprocess the dataset to include only valid annotations
        self.image_ids = self._filter_valid_image_ids()

    def _filter_valid_image_ids(self):
        """
        Filters out image IDs that do not have a valid corresponding image file.

        Returns:
            list: List of valid image IDs.
        """
        valid_image_ids = []
        for image_id in self.coco.imgs.keys():
            image_info = self.coco.loadImgs(image_id)[0]
            image_path = os.path.join(self.root_dir, image_info["file_name"])

            # Check if the image file exists and is valid
            if os.path.exists(image_path) and self._is_valid_image(image_path):
                valid_image_ids.append(image_id)
            else:
                print(f"Invalid or missing image: {image_path} (Image ID: {image_id})")

        return valid_image_ids

    def _is_valid_image(self, image_path):
        """
        Checks if the image at the given path is valid.

        Args:
            image_path (str): Path to the image file.

        Returns:
            bool: True if the image is valid, False otherwise.
        """
        try:
            with Image.open(image_path) as img:
                img.verify()  # Verify that the file is a valid image
            return True
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return False

    def __len__(self):
        """Returns the total number of valid images in the dataset."""
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            image (torch.Tensor): The image as a tensor.
            target (dict): A dictionary containing annotations (bboxes, labels, etc.).
        """
        # Get the image ID
        image_id = self.image_ids[idx]

        # Load image
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info["file_name"])
        image = Image.open(image_path).convert("RGB")  # Load image using PIL and convert to RGB

        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        # Prepare target dictionary
        boxes = []
        labels = []

        for ann in anns:
            # Bounding box in [x_min, y_min, width, height] format
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])  # Convert to [x_min, y_min, x_max, y_max]
            labels.append(ann["category_id"])
 

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Apply transforms (if any)
        if self.transform:
            orig_w, orig_h = image.size
            image = self.transform(image)

            new_h, new_w = image.shape[1:]  # Transformed image dimensions
            scale_x = new_w / orig_w
            scale_y = new_h / orig_h
            boxes[:, [0, 2]] *= scale_x  # Scale x-coordinates
            boxes[:, [1, 3]] *= scale_y  # Scale y-coordinates
        
        target = {
            "boxes": boxes,
            "labels": labels,
        }

        return image, target