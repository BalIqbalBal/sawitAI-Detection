import hydra
from omegaconf import DictConfig, OmegaConf
import optuna
import mlflow
import mlflow.pytorch
import os
import time
import uuid
from train import train  # Import the train function from train.py

@hydra.main(version_base=None, config_path="../config", config_name="config")
def hyperparam_opt(cfg: DictConfig):
    # Auto-generate a unique study name
    timestamp = time.strftime("%Y%m%d_%H%M%S")  # Current timestamp
    unique_id = str(uuid.uuid4())[:8]  # Short unique identifier
    study_name = f"study_{timestamp}_{unique_id}"  # e.g., "study_20231025_143022_abc12345"

    # Create a parent folder named after the model
    model_name = cfg.model.name.lower()  # e.g., "faster_rcnn" or "my_detection_model"
    parent_dir = os.path.join("studies", model_name)
    os.makedirs(parent_dir, exist_ok=True)

    # Create a unique folder for the study under the parent folder
    study_dir = os.path.join(parent_dir, study_name)
    os.makedirs(study_dir, exist_ok=True)

    # Define the objective function for Optuna
    def objective(trial):
        # Get the model-specific hyperparameters from the cfg file
        if "hyperparam_opt" not in cfg:
            raise ValueError(f"No hyperparameter optimization configuration found for model: {model_name}")

        # Dynamically suggest hyperparameters based on the cfg file
        suggested_params = {}
        for param, config in cfg.hyperparam_opt.params.items():
            if config.type == "float":
                suggested_params[param] = trial.suggest_float(param, config.low, config.high, log=config.get("log", False))
            elif config.type == "int":
                suggested_params[param] = trial.suggest_int(param, config.low, config.high)
            elif config.type == "categorical":
                suggested_params[param] = trial.suggest_categorical(param, config.choices)
            else:
                raise ValueError(f"Unsupported parameter type: {config.type}")

        # Update the config with the suggested hyperparameters
        cfg_update = OmegaConf.create(suggested_params)
        OmegaConf.update(cfg, "hyperparam_opt.suggested_params", cfg_update)

        # Start an MLflow run for this trial
        with mlflow.start_run(nested=True):
            # Log hyperparameters
            mlflow.log_params(suggested_params)

            # Train the model and get evaluation metrics
            eval_accuracy = train(cfg)

            # Log the evaluation metric
            mlflow.log_metric("eval_accuracy", eval_accuracy)

            # Return the metric to optimize (e.g., maximize accuracy)
            return eval_accuracy

    # Create an Optuna study
    study = optuna.create_study(
        direction=cfg.hyperparam_opt.direction,  # Maximize or minimize
        study_name=study_name,
        storage=f"sqlite:///{os.path.join(study_dir, 'study.db')}",  # Save study database in the study folder
        load_if_exists=True,
    )

    # Start an MLflow run for the entire optimization process
    with mlflow.start_run():
        # Log the optimization configuration
        mlflow.log_params({
            "optimization_direction": cfg.hyperparam_opt.direction,
            "n_trials": cfg.hyperparam_opt.n_trials,
            "study_name": study_name,
            "model_name": model_name,
        })

        # Run the optimization
        study.optimize(objective, n_trials=cfg.hyperparam_opt.n_trials)

        # Log the best hyperparameters and metric
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_eval_accuracy", study.best_value)

        # Save the study visualization (e.g., optimization history plot)
        try:
            from optuna.visualization import plot_optimization_history
            fig = plot_optimization_history(study)
            fig.write_image(os.path.join(study_dir, "optimization_history.png"))
            mlflow.log_artifact(os.path.join(study_dir, "optimization_history.png"))
        except ImportError:
            print("Plotly is not installed. Skipping visualization.")

        print(f"Best trial:")
        print(f"  Value (eval_accuracy): {study.best_value}")
        print(f"  Params: {study.best_params}")

if __name__ == "__main__":
    hyperparam_opt()