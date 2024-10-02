import os
import argparse
import yaml
import time
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from torch.nn.functional import cross_entropy
from tqdm import tqdm

from models.gpt2_model import GPT2ModelWrapper
from utils.data_utils import CustomDataset
from utils.mlops_utils import setup_evaluation_logging, log_metrics

def compute_perplexity(loss):
    return torch.exp(loss)

def evaluate_model(config):
    # Setup logging
    logger = setup_evaluation_logging(config['mlops'])
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(config['model']['tokenizer_name'])
    
    # Load dataset
    eval_dataset = CustomDataset(
        data_file=config['data']['val_data'],
        tokenizer=tokenizer,
        max_length=config['evaluation']['max_seq_length']
    )
    
    # Create DataLoader
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False
    )
    
    # Initialize model
    model = GPT2ModelWrapper(config_name=config['model']['model_name'])
    model.load_model(config['evaluation']['model_checkpoint'])
    model.to(config['evaluation']['device'])
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    inference_times = []
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(config['evaluation']['device'])
            attention_mask = batch['attention_mask'].to(config['evaluation']['device'])
            labels = batch['labels'].to(config['evaluation']['device'])
            
            start_time = time.time()
            outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            end_time = time.time()
            
            loss = outputs.loss
            logits = outputs.logits
            
            # Compute inference latency
            inference_time = end_time - start_time
            inference_times.append(inference_time)
            
            # Calculate number of correct predictions
            predictions = torch.argmax(logits, dim=-1)
            correct = ((predictions == labels) * attention_mask).sum().item()
            total_correct += correct
            total_tokens += attention_mask.sum().item()
            
            total_loss += loss.item() * input_ids.size(0)
    
    # Compute metrics
    avg_loss = total_loss / len(eval_loader.dataset)
    perplexity = compute_perplexity(torch.tensor(avg_loss))
    accuracy = total_correct / total_tokens
    avg_inference_latency = sum(inference_times) / len(inference_times)
    
    # Log metrics
    metrics = {
        'validation_loss': avg_loss,
        'perplexity': perplexity.item(),
        'accuracy': accuracy,
        'avg_inference_latency': avg_inference_latency
    }
    
    log_metrics(metrics, config['mlops'])
    logger.info(f"Validation Loss: {avg_loss:.4f}")
    logger.info(f"Perplexity: {perplexity.item():.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Average Inference Latency: {avg_inference_latency:.4f} seconds")
    
    # Save metrics to file
    os.makedirs(config['evaluation']['output_dir'], exist_ok=True)
    metrics_file = os.path.join(config['evaluation']['output_dir'], 'evaluation_metrics.yaml')
    with open(metrics_file, 'w') as f:
        yaml.dump(metrics, f)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate GPT-2 Model')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the config file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    evaluate_model(config)
