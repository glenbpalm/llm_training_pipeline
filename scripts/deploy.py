import os
import argparse
import yaml
import torch
from transformers import GPT2Tokenizer
import ray
from ray import serve
from fastapi import FastAPI, Request

from models.gpt2_model import GPT2ModelWrapper
from utils.mlops_utils import setup_deployment_logging

def load_model(model_config, device='cpu'):
    """
    Load the tokenizer and model.

    Args:
        model_config (dict): Configuration dictionary for the model.
        device (str): Device to load the model on ('cpu' or 'cuda').

    Returns:
        model: The loaded GPT-2 model.
        tokenizer: The GPT-2 tokenizer.
    """
    # Load the tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_config['tokenizer_name'])
    model = GPT2ModelWrapper(config_name=model_config['model_name'])
    model.load_model(model_config['model_checkpoint'])
    model.to(device)
    model.eval()
    return model, tokenizer

@serve.deployment(num_replicas=2, route_prefix="/generate")
class GPT2Deployment:
    def __init__(self, model_config, device='cpu', mlops_config=None):
        # Setup logging
        self.logger = setup_deployment_logging(mlops_config)
        self.logger.info("Initializing GPT2Deployment...")

        self.device = device
        self.model, self.tokenizer = load_model(model_config, device)
        self.logger.info("Model and tokenizer loaded successfully.")

    async def __call__(self, request: Request):
        json_input = await request.json()
        prompt = json_input.get('prompt', '')
        max_length = json_input.get('max_length', 50)
        temperature = json_input.get('temperature', 1.0)

        self.logger.info(f"Received request with prompt: '{prompt}'")

        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        with torch.no_grad():
            output = self.model.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                num_return_sequences=1,
                do_sample=True,
            )

        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        self.logger.info(f"Generated text: '{generated_text}'")

        return {'generated_text': generated_text}

def main(config):
    # Setup logging
    logger = setup_deployment_logging(config['mlops'])
    logger.info("Starting deployment...")

    # Initialize Ray
    ray.init()
    serve.start(detached=True)

    # Deploy the model
    GPT2Deployment.deploy(
        model_config=config['model'],
        device=config['deployment']['device'],
        mlops_config=config['mlops']
    )
    logger.info("Model deployed and serving at endpoint '/generate'.")

    # Keep the script running
    import time
    try:
        while True:
            time.sleep(3600)  # Keep alive
    except KeyboardInterrupt:
        logger.info("Shutting down deployment...")
        serve.shutdown()
        ray.shutdown()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deploy GPT-2 Model with Ray Serve')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the config file')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    main(config)
