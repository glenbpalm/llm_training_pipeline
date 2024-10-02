import torch
from torch import nn
from transformers import GPT2LMHeadModel, GPT2Config

class GPT2ModelWrapper(nn.Module):
    def __init__(self, config_name='gpt2', custom_config=None):
        super(GPT2ModelWrapper, self).__init__()
        
        if custom_config:
            self.config = GPT2Config(**custom_config)
            self.model = GPT2LMHeadModel(self.config)
        else:
            self.model = GPT2LMHeadModel.from_pretrained(config_name)
            self.config = self.model.config

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        logits = outputs.logits
        return loss, logits

    def save_model(self, output_dir):
        self.model.save_pretrained(output_dir)
        self.config.save_pretrained(output_dir)

    def load_model(self, model_dir):
        self.model = GPT2LMHeadModel.from_pretrained(model_dir)
        self.config = self.model.config

