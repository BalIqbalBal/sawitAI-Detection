import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split
from dataset.dataset import MyDataset
from dataset.coco_dataset import COCODataset
from dataset.yolo_coco_dataset import YoloCOCODataset  # Import the new YoloDataset
from torchvision import transforms
from hydra.utils import instantiate

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

    # Lookup the dataset class by name
    dataset_class = globals().get(cfg.dataset.name, None)
    if dataset_class is None:
        raise ValueError(f"Dataset {cfg.dataset.name} not supported")
    
    # Initialize the dataset
    if cfg.dataset.name == "YoloCOCODataset":
        dataset = dataset_class(
            root_dir=cfg.dataset.root_dir,
            annotation_file=cfg.dataset.annotation_file,
            input_shape=cfg.dataset.input_shape,
            num_classes=cfg.dataset.num_classes,
            epoch_length=cfg.dataset.epoch_length,
            mosaic=cfg.dataset.mosaic,
            mixup=cfg.dataset.mixup,
            mosaic_prob=cfg.dataset.mosaic_prob,
            mixup_prob=cfg.dataset.mixup_prob,
            train=True,  # Assume training mode for the full dataset
            special_aug_ratio=cfg.dataset.special_aug_ratio,
            transform=train_transform
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