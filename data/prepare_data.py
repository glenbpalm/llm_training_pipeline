import os
import argparse
import random
import json
from pathlib import Path

from tqdm import tqdm
from transformers import GPT2Tokenizer

def load_raw_data(data_dir):
    """
    Load raw text data from the specified directory.
    """
    data = []
    data_dir = Path(data_dir)
    for file_path in data_dir.glob('*.txt'):
        with file_path.open('r', encoding='utf-8') as f:
            text = f.read()
            data.append(text)
    return data

def preprocess_text(text):
    """
    Preprocess the text data (e.g., cleaning, lowercasing).
    """
    # Example preprocessing steps:
    text = text.strip()
    text = text.replace('\n', ' ')
    # Add more preprocessing steps as needed
    return text

def tokenize_data(data, tokenizer):
    """
    Tokenize the data using the specified tokenizer.
    """
    tokenized_data = []
    for text in tqdm(data, desc="Tokenizing data"):
        tokens = tokenizer.encode(text, add_special_tokens=True)
        tokenized_data.append(tokens)
    return tokenized_data

def split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split data into training, validation, and test sets.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    random.shuffle(data)
    total_len = len(data)
    train_end = int(train_ratio * total_len)
    val_end = train_end + int(val_ratio * total_len)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data

def save_data(data, output_path):
    """
    Save the processed data to the specified output path.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)

def main(args):
    # Ensure reproducibility
    random.seed(42)

    # Load raw data
    print("Loading raw data...")
    raw_data = load_raw_data(args.data_dir)
    print(f"Loaded {len(raw_data)} documents.")

    # Preprocess data
    print("Preprocessing data...")
    processed_data = [preprocess_text(text) for text in raw_data]

    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Tokenize data
    print("Tokenizing data...")
    tokenized_data = tokenize_data(processed_data, tokenizer)

    # Split data
    print("Splitting data...")
    train_data, val_data, test_data = split_data(
        tokenized_data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")

    # Save data
    print("Saving data...")
    os.makedirs(args.output_dir, exist_ok=True)
    save_data(train_data, os.path.join(args.output_dir, 'train_data.json'))
    save_data(val_data, os.path.join(args.output_dir, 'val_data.json'))
    save_data(test_data, os.path.join(args.output_dir, 'test_data.json'))

    print("Data preparation completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare data for LLM training.')
    parser.add_argument('--data_dir', type=str, default='data/datasets', help='Directory containing raw data files.')
    parser.add_argument('--output_dir', type=str, default='data/processed', help='Directory to save processed data.')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of training data.')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Ratio of validation data.')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Ratio of test data.')

    args = parser.parse_args()

    main(args)