import unittest
import os
import json
import torch
from transformers import GPT2Tokenizer

from utils.data_utils import CustomDataset

class TestDataUtils(unittest.TestCase):
    def setUp(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = 10

        # Create sample tokenized data
        self.sample_data = [
            self.tokenizer.encode("Hello, world!"),
            self.tokenizer.encode("Testing data utilities."),
            self.tokenizer.encode("This is a sample text for unit testing."),
        ]

        # Save sample data to a JSON file
        self.data_file = 'tests/sample_data.json'
        os.makedirs('tests', exist_ok=True)
        with open(self.data_file, 'w') as f:
            json.dump(self.sample_data, f)

    def test_custom_dataset_length(self):
        dataset = CustomDataset(
            data_file=self.data_file,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        self.assertEqual(len(dataset), len(self.sample_data))

    def test_custom_dataset_item(self):
        dataset = CustomDataset(
            data_file=self.data_file,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        item = dataset[0]
        self.assertIn('input_ids', item)
        self.assertIn('attention_mask', item)
        self.assertIn('labels', item)

        self.assertEqual(len(item['input_ids']), self.max_length)
        self.assertEqual(len(item['attention_mask']), self.max_length)
        self.assertEqual(len(item['labels']), self.max_length)

    def test_padding_and_truncation(self):
        dataset = CustomDataset(
            data_file=self.data_file,
            tokenizer=self.tokenizer,
            max_length=5
        )
        item = dataset[2]  # This sample has more tokens than max_length
        self.assertEqual(len(item['input_ids']), 5)
        self.assertEqual(len(item['attention_mask']), 5)
        self.assertEqual(len(item['labels']), 5)

    def tearDown(self):
        # Clean up
        if os.path.exists(self.data_file):
            os.remove(self.data_file)
        # Note: Be careful with removing directories; ensure it's safe.
        # if os.path.exists('tests'):
        #     os.rmdir('tests')

if __name__ == '__main__':
    unittest.main()
