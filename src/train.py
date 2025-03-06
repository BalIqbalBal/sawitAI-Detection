import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
import numpy as np
import mlflow
import mlflow.pytorch
from tqdm import tqdm
from pathlib import Path
from model.model_factory import get_model
from dataset.data_factory import get_dataset, get_dataloader
from utils.utils import save_model
from utils.metrics import DetMetrics  

def evaluate_model(model, dataloader, device, det_metrics):
    """
    Evaluate the model using DetMetrics.
    """
    model.eval()
    det_metrics.reset()  # Reset metrics for a new evaluation

    with torch.no_grad():
        with tqdm(dataloader, desc="Evaluating") as pbar:
            for i, (images, targets) in enumerate(pbar):
                images = images.to(device)
                outputs = model(images)

                # Convert model outputs to DetMetrics format
                pred_boxes = outputs['boxes'].cpu().numpy()
                pred_scores = outputs['scores'].cpu().numpy()
                pred_labels = outputs['labels'].cpu().numpy()

                # Convert ground truth to DetMetrics format
                gt_boxes = [t['boxes'].cpu().numpy() for t in targets]
                gt_labels = [t['labels'].cpu().numpy() for t in targets]

                # Update DetMetrics with predictions and ground truth
                det_metrics.process(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)

    # Compute and return metrics
    metrics = det_metrics.results_dict
    return metrics

@hydra.main(version_base=None, config_path="../config", config_name="config")
def train(cfg: DictConfig):
    # Set seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    os.makedirs("sementara", exist_ok=True)
    os.makedirs("results/train", exist_ok=True)
    os.makedirs("results/eval", exist_ok=True)
    os.makedirs("results/test", exist_ok=True)

    # Dataset and DataLoader (dynamically loaded)
    train_dataset, test_dataset, eval_dataset = get_dataset(cfg)
    train_dataloader = get_dataloader(cfg, train_dataset)
    test_dataloader = get_dataloader(cfg, test_dataset)
    eval_dataloader = get_dataloader(cfg, eval_dataset)

    # Model (dynamically loaded)
    model = get_model(cfg).to(device)
    print(f"Model: {model}")

    # Instantiate the optimizer
    optimizer = instantiate(cfg.model.optimizer, model.parameters(), lr=cfg.model.optimizer.lr)
    print(f"Optimizer: {optimizer}")

    # Initialize DetMetrics for training, evaluation, and testing
    train_metrics = DetMetrics(save_dir=Path("results/train"), plot=True, names=cfg.dataset.class_names)
    eval_metrics = DetMetrics(save_dir=Path("results/eval"), plot=True, names=cfg.dataset.class_names)
    test_metrics = DetMetrics(save_dir=Path("results/test"), plot=True, names=cfg.dataset.class_names)

    run_name = f"{cfg.model.name}_{cfg.dataset.name}_lr{cfg.model.optimizer.lr}_bs{cfg.dataset.batch_size}"

    # Start MLflow run
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params({
            "epochs": cfg.model.epochs,
            "learning_rate": cfg.model.optimizer.lr,
            "batch_size": cfg.dataset.batch_size,
            "model": cfg.model.name,
            "dataset": cfg.dataset.name,
        })

        # Training loop
        for epoch in range(cfg.model.epochs):
            model.train()
            epoch_loss = 0

            # Reset metrics for the new epoch
            train_metrics.reset()

            # Update epoch_now for YoloDataset (if applicable)
            if hasattr(train_dataset, "epoch_now"):
                train_dataset.epoch_now = epoch
            
            # Use tqdm for progress bar
            with tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{cfg.model.epochs}]") as pbar:
                for batch_idx, (images, targets) in enumerate(pbar):
                    images = images.to(device)
                    targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

                    # Forward pass
                    outputs, loss_dict = model(images, targets)
                    loss = sum(loss for loss in loss_dict.values()) / cfg.dataset.batch_size
                    epoch_loss += loss.item()

                    # Convert model outputs to DetMetrics format
                    pred_boxes = outputs['boxes'].cpu().numpy()
                    pred_scores = outputs['scores'].cpu().numpy()
                    pred_labels = outputs['labels'].cpu().numpy()

                    # Convert ground truth to DetMetrics format
                    gt_boxes = [t['boxes'].cpu().numpy() for t in targets]
                    gt_labels = [t['labels'].cpu().numpy() for t in targets]

                    # Update DetMetrics with predictions and ground truth
                    train_metrics.process(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Update progress bar with current loss
                    pbar.set_postfix({"Loss": loss.item()})
                    
                    # Log loss for each batch
                    mlflow.log_metric("train_loss", loss.item(), step=epoch * len(train_dataloader) + batch_idx)

            # Log average epoch loss
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            mlflow.log_metric("avg_train_loss", avg_epoch_loss, step=epoch)

            # Log training metrics
            train_metrics_results = train_metrics.results_dict
            mlflow.log_metrics({
                "train_mAP": train_metrics_results["metrics/mAP50-95(B)"],
                "train_AP50": train_metrics_results["metrics/mAP50(B)"],
                "train_precision": train_metrics_results["metrics/precision(B)"],
                "train_recall": train_metrics_results["metrics/recall(B)"],
            }, step=epoch)

            # Log training curves (e.g., precision-recall curves)
            train_metrics.log_curves_to_mlflow(prefix="train_")

            # Evaluate on the eval dataset
            eval_metrics.reset()  # Reset metrics for evaluation dataset
            eval_metrics = evaluate_model(model, eval_dataloader, device, eval_metrics)
            
            # Log evaluation metrics
            mlflow.log_metrics({
                "eval_mAP": eval_metrics["metrics/mAP50-95(B)"],
                "eval_AP50": eval_metrics["metrics/mAP50(B)"],
                "eval_precision": eval_metrics["metrics/precision(B)"],
                "eval_recall": eval_metrics["metrics/recall(B)"],
            }, step=epoch)
            
            # Log evaluation curves (e.g., precision-recall curves)
            eval_metrics.log_curves_to_mlflow(prefix="eval_")

            print(f"Epoch [{epoch+1}/{cfg.model.epochs}], "
                  f"Train mAP: {train_metrics_results['metrics/mAP50-95(B)']:.4f}, "
                  f"Eval mAP: {eval_metrics['metrics/mAP50-95(B)']:.4f}")

        # Test the model on the test dataset after training
        test_metrics.reset()  # Reset metrics for test dataset
        test_metrics = evaluate_model(model, test_dataloader, device, test_metrics)
        
        # Log test metrics
        mlflow.log_metrics({
            "test_mAP": test_metrics["metrics/mAP50-95(B)"],
            "test_AP50": test_metrics["metrics/mAP50(B)"],
            "test_precision": test_metrics["metrics/precision(B)"],
            "test_recall": test_metrics["metrics/recall(B)"],
        })
        
        # Log test curves (e.g., precision-recall curves)
        test_metrics.log_curves_to_mlflow(prefix="test_")

        print(f"Test mAP: {test_metrics['metrics/mAP50-95(B)']:.4f}, AP50: {test_metrics['metrics/mAP50(B)']:.4f}")
        
        # Save model and log it as an artifact
        save_model(model, "model.pth")
        mlflow.log_artifact("model.pth")
        
        return test_metrics["metrics/mAP50-95(B)"]

if __name__ == "__main__":
    train()