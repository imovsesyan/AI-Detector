import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from PyPDF2 import PdfReader

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def save_model(model, model_path):
    model.save_pretrained(model_path)

def train_roberta(model, train_dataset, batch_size, epochs, lr, gradient_accumulation_steps, device, model_path):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = len(train_dataloader) * epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
            running_loss += loss.item()
        save_model(model, model_path)

def evaluate_model(model, dataset, batch_size, device, dataset_name):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nEvaluation on {dataset_name}:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, labels=[0, 1], target_names=['Human', 'AI-Generated']))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    return accuracy

def load_trained_model(model_path, tokenizer_path):
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

def extract_text_from_pdfs(pdf_folder):
    texts = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith('.pdf'):
            path = os.path.join(pdf_folder, filename)
            with open(path, 'rb') as f:
                reader = PdfReader(f)
                text = ''
                for page in reader.pages:
                    text += page.extract_text() or ''
                if text.strip():
                    texts.append(text)
    return texts

def main():
    model_path = 'C:/Ai/saved_model'
    tokenizer_path = 'C:/Ai/RoBERTa/bookcorpus_vocab'
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
    model = RobertaForSequenceClassification.from_pretrained(tokenizer_path, num_labels=2)
    train_texts = extract_text_from_pdfs('C:/Ai/human-pdfs') + extract_text_from_pdfs('C:/Ai/ai-pdfs')
    train_labels = [0] * len(extract_text_from_pdfs('C:/Ai/human-pdfs')) + [1] * len(extract_text_from_pdfs('C:/Ai/ai-pdfs'))
    test_texts = extract_text_from_pdfs('C:/Ai/unseen-human-pdfs') + extract_text_from_pdfs('C:/Ai/unseen-ai-pdfs')
    test_labels = [0] * len(extract_text_from_pdfs('C:/Ai/unseen-human-pdfs')) + [1] * len(extract_text_from_pdfs('C:/Ai/unseen-ai-pdfs'))
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_length=512)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_length=512)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    train_roberta(model, train_dataset, batch_size=8, epochs=5, lr=2e-5, gradient_accumulation_steps=2, device=device, model_path=model_path)
    model, tokenizer = load_trained_model(model_path, tokenizer_path)
    evaluate_model(model, test_dataset, batch_size=8, device=device, dataset_name="Unseen Data")
    new_test_texts = extract_text_from_pdfs('C:/Ai/new-human-pdfs') + extract_text_from_pdfs('C:/Ai/new-ai-pdfs')
    new_test_labels = [0] * len(extract_text_from_pdfs('C:/Ai/new-human-pdfs')) + [1] * len(extract_text_from_pdfs('C:/Ai/new-ai-pdfs'))
    new_test_dataset = TextDataset(new_test_texts, new_test_labels, tokenizer, max_length=512)
    evaluate_model(model, new_test_dataset, batch_size=8, device=device, dataset_name="New Unseen Data")

if __name__ == "__main__":
    main()
