import hydra
from omegaconf import DictConfig
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, mean_squared_error, mean_absolute_error, roc_auc_score, confusion_matrix
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import seaborn as sns
from model.model_factory import get_model
from dataset.data_factory import get_dataset, get_dataloader
from utils.utils import load_model

def plot_confusion_matrix(conf_matrix, class_names, filename):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(filename)
    plt.close()

@hydra.main(version_base=None, config_path="../config", config_name="config")
def evaluate(cfg: DictConfig):
    # Device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader (dynamically loaded)
    _, _, eval_dataset = get_dataset(cfg)
    eval_dataloader = get_dataloader(cfg, eval_dataset)

    # Load model (dynamically loaded)
    model = get_model(cfg).to(device)
    load_model(model, "model.pth")

    # Start MLflow run
    with mlflow.start_run():
        # Evaluation
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
        })

        # Plot and log confusion matrix
        class_names = [str(i) for i in range(cfg.model.num_classes)]  # Replace with actual class names if available
        plot_confusion_matrix(conf_matrix, class_names, "eval_confusion_matrix.png")
        mlflow.log_artifact("eval_confusion_matrix.png")

        print(f"Eval Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, AUC: {auc:.4f}")
        print("Eval Confusion Matrix:")
        print(conf_matrix)

if __name__ == "__main__":
    evaluate()