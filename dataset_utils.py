import os
from torch.utils.data import Dataset
from PyPDF2 import PdfReader
import torch

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        encoding = self.tokenizer.encode_plus(
            self.texts[item], max_length=self.max_length, add_special_tokens=True,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[item], dtype=torch.long)
        }

def extract_text_from_pdfs(pdf_folder):
    texts = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith('.pdf'):
            with open(os.path.join(pdf_folder, filename), 'rb') as f:
                reader = PdfReader(f)
                text = "".join([page.extract_text() or '' for page in reader.pages])
                if text.strip():
                    texts.append(text)
    return texts
