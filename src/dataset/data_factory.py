from torch.utils.data import DataLoader, random_split
from dataset.dataset import MyDataset
from dataset.coco_dataset_mopad import COCODataset
from torchvision import transforms

def get_transform(transform_config):
    """
    Dynamically create a transformation pipeline based on the Hydra configuration.

    Args:
        transform_config: List of transformations from the Hydra configuration.

    Returns:
        transform: A composed transformation pipeline.
    """
    transform_list = []
    for transform_item in transform_config:
        transform_name = transform_item.name
        transform_params = {k: v for k, v in transform_item.items() if k != "name"}
        transform_class = getattr(transforms, transform_name)
        transform_list.append(transform_class(**transform_params))
    return transforms.Compose(transform_list)

def get_dataset(cfg):
    """
    Get the dataset based on the configuration.

    Args:
        cfg: Configuration object.

    Returns:
        train_dataset, test_dataset, eval_dataset: Split datasets with transformations applied.
    """
    # Get transformations for train, test, and eval sets
    train_transform = get_transform(cfg.dataset.transform.train) if "transform" in cfg.dataset and "train" in cfg.dataset.transform else None
    test_transform = get_transform(cfg.dataset.transform.test) if "transform" in cfg.dataset and "test" in cfg.dataset.transform else None
    eval_transform = get_transform(cfg.dataset.transform.eval) if "transform" in cfg.dataset and "eval" in cfg.dataset.transform else None

    if cfg.dataset.name == "MyDataset":
        train_dataset = MyDataset(root_dir=cfg.dataset.root_dir, transform=train_transform)
        test_dataset = MyDataset(root_dir=cfg.dataset.root_dir, transform=test_transform)
        eval_dataset = MyDataset(root_dir=cfg.dataset.root_dir, transform=eval_transform)
    elif cfg.dataset.name == "COCODataset":
        train_dataset = COCODataset(root_dir=cfg.dataset.root_dir, annotation_file=cfg.dataset.annotation_file, transform=train_transform)
        test_dataset = COCODataset(root_dir=cfg.dataset.root_dir, annotation_file=cfg.dataset.annotation_file, transform=test_transform)
        eval_dataset = COCODataset(root_dir=cfg.dataset.root_dir, annotation_file=cfg.dataset.annotation_file, transform=eval_transform)
    else:
        raise ValueError(f"Dataset {cfg.dataset.name} not supported")

    # Split dataset into train, test, and eval
    train_size = int(cfg.dataset.train_ratio * len(train_dataset))
    test_size = int(cfg.dataset.test_ratio * len(train_dataset))
    eval_size = len(train_dataset) - train_size - test_size

    train_dataset, test_dataset, eval_dataset = random_split(
        train_dataset, [train_size, test_size, eval_size]
    )

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
    )