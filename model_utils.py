import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

def load_model(model_path, tokenizer_path, num_labels=2):
    model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

def save_model(model, model_path):
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

