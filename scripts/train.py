import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import ray
from ray import tune
from ray.tune.integration.torch import DistributedTrainableCreator

from models.gpt2_model import GPT2ModelWrapper
from utils.data_utils import CustomDataset
from utils.mlops_utils import setup_training_logging, log_metrics
from utils.distributed_utils import init_distributed_mode, cleanup

def train_model(config):
    # Initialize distributed training
    init_distributed_mode(config['distributed'])

    # Setup logging
    logger = setup_training_logging(config['mlops'])

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(config['model']['tokenizer_name'])

    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    train_dataset = CustomDataset(
        data_file=config['data']['train_data'],
        tokenizer=tokenizer,
        max_length=config['training']['max_seq_length']
    )

    # Create DataLoader
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=train_sampler
    )

    # Initialize model
    model = GPT2ModelWrapper(config_name=config['model']['model_name'])
    model.to(config['training']['device'])

    # Wrap model for distributed training
    model = DDP(model, device_ids=[config['training']['local_rank']])

    # Set up optimizer
    optimizer = AdamW(model.parameters(), lr=config['training']['learning_rate'])

    # Training loop
    for epoch in range(config['training']['epochs']):
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(config['training']['device'])
            attention_mask = batch['attention_mask'].to(config['training']['device'])
            labels = batch['labels'].to(config['training']['device'])

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Calculate average loss
        avg_loss = total_loss / len(train_loader)

        # Log metrics
        if dist.get_rank() == 0:
            log_metrics({'epoch': epoch, 'loss': avg_loss}, config['mlops'])
            print(f"Epoch [{epoch+1}/{config['training']['epochs']}], Loss: {avg_loss:.4f}")

    # Save model checkpoint (only from the main process)
    if dist.get_rank() == 0:
        os.makedirs(config['training']['output_dir'], exist_ok=True)
        model.module.save_model(config['training']['output_dir'])

    # Clean up distributed processes
    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distributed GPT-2 Training')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the config file')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize Ray
    ray.init()

    # Create a distributed trainable
    DistributedTrainable = DistributedTrainableCreator(
        train_model,
        num_workers=config['distributed']['num_workers'],
        num_cpus_per_worker=config['distributed']['num_cpus_per_worker'],
        use_gpu=config['distributed']['use_gpu'],
        backend=config['distributed'].get('backend', 'nccl')  # Use 'gloo' if you're training on CPU
    )

    # Run training
    analysis = tune.run(
        DistributedTrainable,
        config={'config': config},
        num_samples=1
    )

    # Shutdown Ray
    ray.shutdown()
