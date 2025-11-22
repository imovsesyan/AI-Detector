import os
import time
import re
import matplotlib.pyplot as plt
os.environ["WANDB_MODE"] = "offline"

import torch
from torch.optim import AdamW
from transformers import (
    get_linear_schedule_with_warmup,
    RobertaForSequenceClassification,
    RobertaTokenizer,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from dataset_utils import TextDataset, extract_text_from_pdfs

try:
    from torch.amp import GradScaler, autocast
    autocast_device = 'cuda'
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    autocast_device = 'cuda'

def clean_armenian_text(text_list):
    pattern = re.compile(r'[^Ա-ևա-և\s.,!?«»-]')
    return [pattern.sub('', text) for text in text_list]

def train(model, train_dataset, val_dataset, batch_size, epochs, lr, device, model_path, patience=5):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if len(val_loader) == 0:
        print("Validation loader is empty. Aborting training.")
        return

    class_counts = torch.bincount(torch.tensor(train_dataset.labels))
    class_weights = 1. / class_counts.float()
    weights = class_weights[train_dataset.labels]
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    scaler = GradScaler() if torch.cuda.is_available() else None
    model.to(device)

    best_val_loss = float('inf')
    early_stop_counter = 0

    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]")
        for batch in progress_bar:
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items()}
            inputs['labels'] = inputs.pop('label')

            if scaler:
                with autocast(device_type=autocast_device):
                    outputs = model(**inputs)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                outputs = model(**inputs)
                loss = outputs.loss
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_accuracy)

        end_time = time.time()
        print(f"Epoch {epoch+1} completed in {end_time - start_time:.2f} seconds")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            model.save_pretrained(model_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                break

    print("\nEvaluating best saved model:")
    best_model = RobertaForSequenceClassification.from_pretrained(model_path).to(device)
    evaluate(best_model, val_loader, device)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Loss & Validation Accuracy")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("training_plot.png")
    plt.show()

def evaluate(model, dataloader, device):
    if len(dataloader) == 0:
        print("Validation dataloader is empty. Skipping evaluation.")
        return float('inf'), 0.0

    model.eval()
    loss_total = 0.0
    preds = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = {k: v.to(device) for k, v in batch.items()}
            inputs['labels'] = inputs.pop('label')
            outputs = model(**inputs)
            loss_total += outputs.loss.item()
            predictions = torch.argmax(outputs.logits, dim=1)
            preds.extend(predictions.cpu().numpy())
            labels.extend(inputs['labels'].cpu().numpy())

    avg_loss = loss_total / len(dataloader)
    accuracy = accuracy_score(labels, preds)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(labels, preds, target_names=["Human", "AI-Generated"], zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))
    return avg_loss, accuracy

def load_model_and_tokenizer(model_path, tokenizer_path, num_labels=2):
    model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

if __name__ == "__main__":
    base_model_path = "C:/Ai/RoBERTa/bookcorpus_vocab"
    model, tokenizer = load_model_and_tokenizer(base_model_path, base_model_path)

    human_texts = clean_armenian_text(extract_text_from_pdfs('C:/Ai/human-pdfs'))
    ai_texts = clean_armenian_text(extract_text_from_pdfs('C:/Ai/ai-pdfs'))
    unseen_human_texts = clean_armenian_text(extract_text_from_pdfs('C:/Ai/unseen-human-pdfs'))
    unseen_ai_texts = clean_armenian_text(extract_text_from_pdfs('C:/Ai/unseen-ai-pdfs'))

    texts = human_texts + ai_texts
    labels = [0] * len(human_texts) + [1] * len(ai_texts)

    unseen_texts = unseen_human_texts + unseen_ai_texts
    unseen_labels = [0] * len(unseen_human_texts) + [1] * len(unseen_ai_texts)

    if not unseen_texts:
        print("No unseen validation data available. Exiting.")
        exit()

    train_dataset = TextDataset(texts, labels, tokenizer, max_length=512)
    val_dataset = TextDataset(unseen_texts, unseen_labels, tokenizer, max_length=512)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train(
        model,
        train_dataset,
        val_dataset,
        batch_size=16,
        epochs=10,
        lr=2e-5,
        device=device,
        model_path="C:/Ai/RoBERTa/best_model",
        patience=5
    )
