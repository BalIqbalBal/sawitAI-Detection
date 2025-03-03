import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
from torch import nn, optim
from sklearn.metrics import accuracy_score, f1_score, recall_score, mean_squared_error, mean_absolute_error, roc_auc_score, confusion_matrix
import numpy as np
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import seaborn as sns
from model.model_factory import get_model
from dataset.data_factory import get_dataset, get_dataloader
from utils.utils import save_model

def plot_confusion_matrix(conf_matrix, class_names, filename):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(filename)
    plt.close()

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
            for batch_idx, (images, targets) in enumerate(train_dataloader):
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

                if cfg.model.name == "FasterRCNN":
                    loss_dict = model(images, targets)
                    loss = sum(loss for loss in loss_dict.values())
                else:
                    # Forward pass
                    output = model(images)
                    criterion = instantiate(cfg.model.loss_fn)
                    print(f"Loss function: {criterion}")
                    loss = criterion(output, target)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Logging
                if batch_idx % 10 == 0:
                    print(f"Epoch [{epoch+1}/{cfg.model.epochs}], Batch [{batch_idx}/{len(train_dataloader)}], Loss: {loss.item():.4f}")
                    mlflow.log_metric("train_loss", loss.item(), step=epoch * len(train_dataloader) + batch_idx)

            # Evaluation on the eval dataset after each epoch
            model.eval()
            all_targets = []
            all_predictions = []
            with torch.no_grad():
                for data, target in eval_dataloader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    predictions = torch.argmax(output, dim=1)
                    all_targets.extend(target.cpu().numpy())
                    all_predictions.extend(predictions.cpu().numpy())

            # Metrics
            accuracy = accuracy_score(all_targets, all_predictions)
            f1 = f1_score(all_targets, all_predictions, average="weighted")
            recall = recall_score(all_targets, all_predictions, average="weighted")
            mse = mean_squared_error(all_targets, all_predictions)
            mae = mean_absolute_error(all_targets, all_predictions)
            auc = roc_auc_score(all_targets, all_predictions)
            conf_matrix = confusion_matrix(all_targets, all_predictions)

            # Log metrics
            mlflow.log_metrics({
                "eval_accuracy": accuracy,
                "eval_f1": f1,
                "eval_recall": recall,
                "eval_mse": mse,
                "eval_mae": mae,
                "eval_auc": auc,
            }, step=epoch)

            # Plot and log confusion matrix
            class_names = [str(i) for i in range(cfg.model.num_classes)]  # Replace with actual class names if available
            plot_confusion_matrix(conf_matrix, class_names, "confusion_matrix.png")
            mlflow.log_artifact("confusion_matrix.png")

            print(f"Epoch [{epoch+1}/{cfg.model.epochs}], Eval Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, AUC: {auc:.4f}")
            print("Eval Confusion Matrix:")
            print(conf_matrix)

        # Test the model on the test dataset after training
        model.eval()
        all_targets = []
        all_predictions = []
        with torch.no_grad():
            for data, target in test_dataloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                predictions = torch.argmax(output, dim=1)
                all_targets.extend(target.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

        # Metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions, average="weighted")
        recall = recall_score(all_targets, all_predictions, average="weighted")
        mse = mean_squared_error(all_targets, all_predictions)
        mae = mean_absolute_error(all_targets, all_predictions)
        auc = roc_auc_score(all_targets, all_predictions)
        conf_matrix = confusion_matrix(all_targets, all_predictions)

        # Log test metrics
        mlflow.log_metrics({
            "test_accuracy": accuracy,
            "test_f1": f1,
            "test_recall": recall,
            "test_mse": mse,
            "test_mae": mae,
            "test_auc": auc,
        })

        # Plot and log test confusion matrix
        plot_confusion_matrix(conf_matrix, class_names, "test_confusion_matrix.png")
        mlflow.log_artifact("test_confusion_matrix.png")

        print(f"Test Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, AUC: {auc:.4f}")
        print("Test Confusion Matrix:")
        print(conf_matrix)

        # Save model and log it as an artifact
        save_model(model, "model.pth", cfg.model.name)
        mlflow.log_artifact("model.pth")

        return accuracy

if __name__ == "__main__":
    train()