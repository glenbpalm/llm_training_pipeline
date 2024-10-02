import json
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length):
        """
        Custom Dataset for loading tokenized text data.

        Args:
            data_file (str): Path to the JSON file containing the tokenized data.
            tokenizer (PreTrainedTokenizer): Tokenizer to process the data.
            max_length (int): Maximum sequence length for padding/truncating.
        """
        # Load the data
        with open(data_file, 'r') as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve the tokenized input IDs
        input_ids = self.data[idx]

        # Truncate or pad the input IDs to max_length
        input_ids = input_ids[:self.max_length]
        padding_length = self.max_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length

        # Create attention mask
        attention_mask = [1] * len(input_ids)
        if padding_length > 0:
            attention_mask[-padding_length:] = [0] * padding_length

        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = input_ids.clone()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
