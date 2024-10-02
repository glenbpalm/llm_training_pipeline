# mlops_utils.py

import logging
import os

def setup_training_logging(mlops_config):
    """
    Sets up logging and initializes MLflow or Weights & Biases tracking for training.

    Args:
        mlops_config (dict): Configuration dictionary for MLOps.
            - 'tracking_uri': URI for the tracking server.
            - 'experiment_name': Name of the experiment.
            - 'log_dir': Directory to save log files.
            - 'use_mlflow': Whether to use MLflow for tracking.
            - 'use_wandb': Whether to use Weights & Biases for tracking.

    Returns:
        logger: Configured logger object.
    """
    log_dir = mlops_config.get('log_dir', 'logs/')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'training.log')

    # Set up logging to file
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers to the logger
    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Also log to console
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # Initialize MLflow or Weights & Biases
    if mlops_config.get('use_mlflow', False):
        import mlflow
        if mlflow.active_run() is None:
            mlflow.set_tracking_uri(mlops_config['tracking_uri'])
            mlflow.set_experiment(mlops_config['experiment_name'])
            mlflow.start_run(run_name='training')
        logger.info("MLflow tracking initialized for training.")
    elif mlops_config.get('use_wandb', False):
        import wandb
        if wandb.run is None:
            wandb.init(project=mlops_config['experiment_name'], name='training')
        logger.info("Weights & Biases tracking initialized for training.")

    return logger

def setup_evaluation_logging(mlops_config):
    """
    Sets up logging and initializes MLflow or Weights & Biases tracking for evaluation.

    Args:
        mlops_config (dict): Configuration dictionary for MLOps.

    Returns:
        logger: Configured logger object.
    """
    log_dir = mlops_config.get('log_dir', 'logs/')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'evaluation.log')

    # Set up logging to file
    logger = logging.getLogger('evaluation')
    logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers to the logger
    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Also log to console
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # Initialize MLflow or Weights & Biases
    if mlops_config.get('use_mlflow', False):
        import mlflow
        if mlflow.active_run() is None:
            mlflow.set_tracking_uri(mlops_config['tracking_uri'])
            mlflow.set_experiment(mlops_config['experiment_name'])
            mlflow.start_run(run_name='evaluation')
        logger.info("MLflow tracking initialized for evaluation.")
    elif mlops_config.get('use_wandb', False):
        import wandb
        if wandb.run is None:
            wandb.init(project=mlops_config['experiment_name'], name='evaluation')
        logger.info("Weights & Biases tracking initialized for evaluation.")

    return logger

def setup_deployment_logging(mlops_config):
    """
    Sets up logging for deployment.

    Args:
        mlops_config (dict): Configuration dictionary for MLOps.

    Returns:
        logger: Configured logger object.
    """
    log_dir = mlops_config.get('log_dir', 'logs/')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'deployment.log')

    # Set up logging to file
    logger = logging.getLogger('deployment')
    logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers to the logger
    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Also log to console
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    logger.info("Deployment logging initialized.")

    return logger

def log_metrics(metrics, mlops_config, phase='training'):
    """
    Logs metrics to MLflow or Weights & Biases and the logger.

    Args:
        metrics (dict): Dictionary of metrics to log.
        mlops_config (dict): Configuration dictionary for MLOps.
        phase (str): 'training', 'evaluation', or 'deployment'.
    """
    logger = logging.getLogger(phase)

    # Log metrics to MLflow or Weights & Biases
    if mlops_config.get('use_mlflow', False):
        import mlflow
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
    elif mlops_config.get('use_wandb', False):
        import wandb
        wandb.log(metrics)

    # Additionally, log metrics using the logger
    for key, value in metrics.items():
        logger.info(f"{key}: {value}")

def end_run(mlops_config):
    """
    Ends the MLflow or Weights & Biases run.

    Args:
        mlops_config (dict): Configuration dictionary for MLOps.
    """
    if mlops_config.get('use_mlflow', False):
        import mlflow
        if mlflow.active_run() is not None:
            mlflow.end_run()
    elif mlops_config.get('use_wandb', False):
        import wandb
        if wandb.run is not None:
            wandb.finish()
