import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from model_utils import save_model
from dataset_utils import TextDataset, extract_text_from_pdfs
from transformers import RobertaForSequenceClassification, RobertaTokenizer

def evaluate(model, dataset, batch_size, device, dataset_name):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    model.to(device)
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['label'].cpu().numpy())

    print(f"\nEvaluation on {dataset_name}:")
    print(f"Accuracy: {accuracy_score(all_labels, all_preds) * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, labels=[0, 1], target_names=['Human', 'AI-Generated']))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

def load_model_and_tokenizer(model_path, tokenizer_path, num_labels):
    model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

if __name__ == "__main__":
    model_path = "C:/Ai/RoBERTa/bookcorpus_vocab"
    tokenizer_path = "C:/Ai/RoBERTa/bookcorpus_vocab"
    
    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path, num_labels=2)

    test_texts = extract_text_from_pdfs('C:/Ai/new-human-pdfs') + extract_text_from_pdfs('C:/Ai/new-ai-pdfs')
    test_labels = [0] * len(extract_text_from_pdfs('C:/Ai/new-human-pdfs')) + [1] * len(extract_text_from_pdfs('C:/Ai/new-ai-pdfs'))
    
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_length=512)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate(model, test_dataset, batch_size=8, device=device, dataset_name="New Unseen Data")
