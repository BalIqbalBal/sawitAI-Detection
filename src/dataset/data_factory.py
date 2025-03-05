import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split
from dataset.dataset import MyDataset
from dataset.coco_dataset import COCODataset
from dataset.yolo_coco_dataset import YoloCOCODataset  # Import the new YoloDataset
from torchvision import transforms
from hydra.utils import instantiate

# Mapping of dataset names to classes
DATASET_MAPPING = {
    "MyDataset": MyDataset,
    "COCODataset": COCODataset,
    "YoloCOCODataset": YoloCOCODataset,
}

def yolo_collate_fn(batch):
    """
    Collate function for YOLO dataset.
    Args:
        batch: List of tuples (image, labels_out).
    Returns:
        images: Tensor of shape (batch_size, C, H, W).
        bboxes: Tensor of shape (N, 6), where N is the total number of boxes,
                and each row contains [batch_index, class_id, center_x, center_y, width, height].
    """
    images = []
    bboxes = []

    for i, (img, label) in enumerate(batch):
        images.append(img)

        if label.shape[0] > 0:  # Check if there are any boxes
            # Add batch index to each box
            batch_index = torch.full((label.shape[0], 1), i)
            combined = torch.cat((batch_index, label[:, 1:]), dim=1)
            bboxes.append(combined)

    # Stack images into a single tensor
    images = torch.stack(images, dim=0)

    # Concatenate all bounding boxes into a single tensor
    if bboxes:
        bboxes = torch.cat(bboxes, dim=0)
    else:
        bboxes = torch.zeros((0, 6), dtype=torch.float32)

    return images, bboxes

def collate_fn(batch):
    return tuple(zip(*batch))

def get_transform(transform_config):
    """
    Dynamically create a transformation pipeline based on the Hydra configuration.

    Args:
        transform_config: List of transformations from the Hydra configuration.

    Returns:
        transform: A composed transformation pipeline.
    """
    transform_list = [instantiate(transform_item) for transform_item in transform_config]
    return transforms.Compose(transform_list)

def get_dataset(cfg):
    """
    Get the dataset based on the configuration.
    Args:
        cfg: Configuration object.
    Returns:
        train_dataset, test_dataset, eval_dataset: Split datasets with transformations applied.
    """
    train_transform = get_transform(cfg.dataset.transform.train) if "train" in cfg.dataset.transform else None
    test_transform = get_transform(cfg.dataset.transform.test) if "test" in cfg.dataset.transform else None
    eval_transform = get_transform(cfg.dataset.transform.eval) if "eval" in cfg.dataset.transform else None

    # Get the dataset class from mapping
    dataset_class = DATASET_MAPPING.get(cfg.dataset.name)
    if dataset_class is None:
        raise ValueError(f"Dataset {cfg.dataset.name} not supported")
    
    # Initialize the datasets
    if cfg.dataset.name == "YoloCOCODataset":
        train_dataset = dataset_class(
            root_dir=cfg.dataset.root_dir,
            annotation_file=cfg.dataset.annotation_file,
            input_shape=cfg.dataset.input_shape,
            num_classes=cfg.dataset.num_classes,
            epoch_length=cfg.dataset.epoch_length,
            mosaic=cfg.dataset.mosaic,
            mixup=cfg.dataset.mixup,
            mosaic_prob=cfg.dataset.mosaic_prob,
            mixup_prob=cfg.dataset.mixup_prob,
            train=True,  # Training mode
            special_aug_ratio=cfg.dataset.special_aug_ratio,
            transform=train_transform
        )
        test_dataset = dataset_class(
            root_dir=cfg.dataset.root_dir,
            annotation_file=cfg.dataset.annotation_file,
            input_shape=cfg.dataset.input_shape,
            num_classes=cfg.dataset.num_classes,
            epoch_length=cfg.dataset.epoch_length,
            mosaic=False,  # Disable mosaic for testing
            mixup=False,  # Disable mixup for testing
            mosaic_prob=0.0,
            mixup_prob=0.0,
            train=False,  # Evaluation mode
            special_aug_ratio=0.0,
            transform=test_transform
        )
        eval_dataset = dataset_class(
            root_dir=cfg.dataset.root_dir,
            annotation_file=cfg.dataset.annotation_file,
            input_shape=cfg.dataset.input_shape,
            num_classes=cfg.dataset.num_classes,
            epoch_length=cfg.dataset.epoch_length,
            mosaic=False,  # Disable mosaic for evaluation
            mixup=False,  # Disable mixup for evaluation
            mosaic_prob=0.0,
            mixup_prob=0.0,
            train=False,  # Evaluation mode
            special_aug_ratio=0.0,
            transform=eval_transform
        )
    else:
        dataset = dataset_class(
            root_dir=cfg.dataset.root_dir,
            annotation_file=cfg.dataset.annotation_file,
            transform=train_transform
        )
        # Split the dataset
        train_size = int(cfg.dataset.train_ratio * len(dataset))
        test_size = int(cfg.dataset.test_ratio * len(dataset))
        eval_size = len(dataset) - train_size - test_size
        train_dataset, test_dataset, eval_dataset = random_split(dataset, [train_size, test_size, eval_size])
    
    return train_dataset, test_dataset, eval_dataset

def get_dataloader(cfg, dataset):
    """
    Get the DataLoader for the dataset.
    Args:
        cfg: Configuration object.
        dataset: Dataset to create the DataLoader for.
    Returns:
        DataLoader: DataLoader for the dataset.
    """
    return DataLoader(
        dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=cfg.dataset.shuffle,
        num_workers=cfg.dataset.num_workers,
        collate_fn=collate_fn
    )