import unittest
import os
import logging

from utils.mlops_utils import (
    setup_training_logging,
    setup_evaluation_logging,
    setup_deployment_logging,
    log_metrics,
    end_run
)

class TestMLOpsUtils(unittest.TestCase):
    def setUp(self):
        self.mlops_config = {
            'tracking_uri': 'http://localhost:5000',
            'experiment_name': 'test_experiment',
            'log_dir': 'tests/logs/',
            'use_mlflow': False,
            'use_wandb': False
        }
        os.makedirs(self.mlops_config['log_dir'], exist_ok=True)

    def test_setup_training_logging(self):
        logger = setup_training_logging(self.mlops_config)
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, 'training')

        # Check if log file is created
        log_file = os.path.join(self.mlops_config['log_dir'], 'training.log')
        self.assertTrue(os.path.exists(log_file))

    def test_setup_evaluation_logging(self):
        logger = setup_evaluation_logging(self.mlops_config)
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, 'evaluation')

        # Check if log file is created
        log_file = os.path.join(self.mlops_config['log_dir'], 'evaluation.log')
        self.assertTrue(os.path.exists(log_file))

    def test_setup_deployment_logging(self):
        logger = setup_deployment_logging(self.mlops_config)
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, 'deployment')

        # Check if log file is created
        log_file = os.path.join(self.mlops_config['log_dir'], 'deployment.log')
        self.assertTrue(os.path.exists(log_file))

    def test_log_metrics(self):
        # Set up a logger
        logger = setup_training_logging(self.mlops_config)
        metrics = {'accuracy': 0.95, 'loss': 0.1}
        log_metrics(metrics, self.mlops_config, phase='training')

        # Check if metrics are logged
        log_file = os.path.join(self.mlops_config['log_dir'], 'training.log')
        with open(log_file, 'r') as f:
            logs = f.read()
            self.assertIn('accuracy: 0.95', logs)
            self.assertIn('loss: 0.1', logs)

    def tearDown(self):
        # Clean up log files and directory
        log_files = [
            'training.log',
            'evaluation.log',
            'deployment.log'
        ]
        for log_file in log_files:
            path = os.path.join(self.mlops_config['log_dir'], log_file)
            if os.path.exists(path):
                os.remove(path)
        # Remove logs directory if empty
        if os.path.exists(self.mlops_config['log_dir']):
            try:
                os.rmdir(self.mlops_config['log_dir'])
            except OSError:
                pass  # Directory not empty

if __name__ == '__main__':
    unittest.main()
