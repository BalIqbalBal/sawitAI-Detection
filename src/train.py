import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
import numpy as np
import mlflow
import mlflow.pytorch
from model.model_factory import get_model
from dataset.data_factory import get_dataset, get_dataloader
from utils.utils import save_model
from utils.coco_utils import convert_to_coco_format, evaluate_detections, plot_precision_recall_curve

@hydra.main(version_base=None, config_path="../config", config_name="config")
def train(cfg: DictConfig):
    # Set seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

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

    # Start MLflow run
    with mlflow.start_run():
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
            
            for batch_idx, (images, targets) in enumerate(train_dataloader):
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

                # Forward pass
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
                epoch_loss += loss.item()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Logging
                if batch_idx % 10 == 0:
                    print(f"Epoch [{epoch+1}/{cfg.model.epochs}], Batch [{batch_idx}/{len(train_dataloader)}], Loss: {loss.item():.4f}")
                    mlflow.log_metric("train_loss", loss.item(), step=epoch * len(train_dataloader) + batch_idx)

            # Log average epoch loss
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            mlflow.log_metric("avg_train_loss", avg_epoch_loss, step=epoch)
            
            # Evaluation on the eval dataset after each epoch
            model.eval()
            all_predictions = []
            all_targets = []
            image_ids = []
            
            with torch.no_grad():
                for i, (images, targets) in enumerate(eval_dataloader):
                    images = list(img.to(device) for img in images)
                    outputs = model(images)
                    
                    # Store predictions and targets
                    all_predictions.extend(outputs)
                    all_targets.extend(targets)
                    image_ids.extend(range(i * len(images), (i + 1) * len(images)))
            
            # Convert to COCO format and evaluate
            coco_gt, coco_dt = convert_to_coco_format(all_predictions, all_targets, image_ids)
            metrics, pr_curves = evaluate_detections(coco_gt, coco_dt)
            
            # Log metrics
            mlflow.log_metrics({
                "eval_mAP": metrics["mAP"],
                "eval_AP50": metrics["AP50"],
                "eval_AP75": metrics["AP75"],
                "eval_AP_small": metrics["AP_small"],
                "eval_AP_medium": metrics["AP_medium"],
                "eval_AP_large": metrics["AP_large"],
                "eval_AR_max1": metrics["AR_max1"],
                "eval_AR_max10": metrics["AR_max10"],
                "eval_AR_max100": metrics["AR_max100"],
            }, step=epoch)
            
            # Plot and log precision-recall curves
            for cat_id, (recalls, precisions) in pr_curves.items():
                ap = np.mean(precisions)
                filename = f"pr_curve_cat{cat_id}_epoch{epoch}.png"
                plot_precision_recall_curve(precisions, recalls, ap, filename)
                mlflow.log_artifact(filename)
            
            print(f"Epoch [{epoch+1}/{cfg.model.epochs}], Eval mAP: {metrics['mAP']:.4f}, AP50: {metrics['AP50']:.4f}, AP75: {metrics['AP75']:.4f}")

        # Test the model on the test dataset after training
        model.eval()
        all_predictions = []
        all_targets = []
        image_ids = []
        
        with torch.no_grad():
            for i, (images, targets) in enumerate(test_dataloader):
                images = list(img.to(device) for img in images)
                outputs = model(images)
                
                # Store predictions and targets
                all_predictions.extend(outputs)
                all_targets.extend(targets)
                image_ids.extend(range(i * len(images), (i + 1) * len(images)))
        
        # Convert to COCO format and evaluate
        coco_gt, coco_dt = convert_to_coco_format(all_predictions, all_targets, image_ids)
        metrics, pr_curves = evaluate_detections(coco_gt, coco_dt)
        
        # Log test metrics
        mlflow.log_metrics({
            "test_mAP": metrics["mAP"],
            "test_AP50": metrics["AP50"],
            "test_AP75": metrics["AP75"],
            "test_AP_small": metrics["AP_small"],
            "test_AP_medium": metrics["AP_medium"],
            "test_AP_large": metrics["AP_large"],
            "test_AR_max1": metrics["AR_max1"],
            "test_AR_max10": metrics["AR_max10"],
            "test_AR_max100": metrics["AR_max100"],
        })
        
        # Plot and log test precision-recall curves
        for cat_id, (recalls, precisions) in pr_curves.items():
            ap = np.mean(precisions)
            filename = f"test_pr_curve_cat{cat_id}.png"
            plot_precision_recall_curve(precisions, recalls, ap, filename)
            mlflow.log_artifact(filename)
        
        print(f"Test mAP: {metrics['mAP']:.4f}, AP50: {metrics['AP50']:.4f}, AP75: {metrics['AP75']:.4f}")
        
        # Save model and log it as an artifact
        save_model(model, "model.pth", cfg.model.name)
        mlflow.log_artifact("model.pth")
        
        return metrics["mAP"]

if __name__ == "__main__":
    train()