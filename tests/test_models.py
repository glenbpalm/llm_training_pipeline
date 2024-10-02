import unittest
import torch
from transformers import GPT2Tokenizer

from models.gpt2_model import GPT2ModelWrapper

class TestGPT2ModelWrapper(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = 'gpt2'
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = GPT2ModelWrapper(config_name=self.model_name)
        self.model.to(self.device)
        self.model.eval()

        # Sample input
        self.sample_text = "Hello, how are you?"
        self.input_ids = self.tokenizer.encode(self.sample_text, return_tensors='pt').to(self.device)
        self.attention_mask = torch.ones_like(self.input_ids).to(self.device)
        self.labels = self.input_ids.clone()

    def test_forward_pass(self):
        with torch.no_grad():
            loss, logits = self.model(
                input_ids=self.input_ids,
                attention_mask=self.attention_mask,
                labels=self.labels
            )
        self.assertIsNotNone(loss)
        self.assertIsNotNone(logits)
        self.assertEqual(logits.shape[0], self.input_ids.shape[0])  # Batch size
        self.assertEqual(logits.shape[1], self.input_ids.shape[1])  # Sequence length
        self.assertEqual(logits.shape[2], self.model.config.vocab_size)  # Vocabulary size

    def test_save_and_load_model(self):
        # Save the model
        output_dir = 'tests/test_checkpoint'
        self.model.save_model(output_dir)

        # Load the model
        loaded_model = GPT2ModelWrapper()
        loaded_model.load_model(output_dir)
        loaded_model.to(self.device)
        loaded_model.eval()

        # Check that the loaded model produces the same output
        with torch.no_grad():
            original_loss, original_logits = self.model(
                input_ids=self.input_ids,
                attention_mask=self.attention_mask,
                labels=self.labels
            )
            loaded_loss, loaded_logits = loaded_model(
                input_ids=self.input_ids,
                attention_mask=self.attention_mask,
                labels=self.labels
            )

        self.assertTrue(torch.allclose(original_logits, loaded_logits, atol=1e-6))

    def test_custom_config(self):
        # Create a custom configuration
        custom_config = {
            'vocab_size': 50257,
            'n_positions': 1024,
            'n_ctx': 1024,
            'n_embd': 256,  # Smaller embedding size
            'n_layer': 4,   # Fewer layers
            'n_head': 4,    # Fewer attention heads
        }
        custom_model = GPT2ModelWrapper(custom_config=custom_config)
        custom_model.to(self.device)
        custom_model.eval()

        # Check model configuration
        self.assertEqual(custom_model.config.n_embd, 256)
        self.assertEqual(custom_model.config.n_layer, 4)
        self.assertEqual(custom_model.config.n_head, 4)

        # Forward pass
        with torch.no_grad():
            loss, logits = custom_model(
                input_ids=self.input_ids,
                attention_mask=self.attention_mask,
                labels=self.labels
            )
        self.assertIsNotNone(loss)
        self.assertIsNotNone(logits)

if __name__ == '__main__':
    unittest.main()
