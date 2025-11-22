import os
os.environ["WANDB_MODE"] = "offline"

import torch
from torch.optim import AdamW
from transformers import (
    get_linear_schedule_with_warmup,
    RobertaForSequenceClassification,
    RobertaTokenizer,
)
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from dataset_utils import TextDataset, extract_text_from_pdfs
from torch.amp import GradScaler, autocast

def train(model, train_dataset, val_dataset, batch_size, epochs, lr, device, model_path, patience=2):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    scaler = GradScaler('cuda')
    model.to(device)

    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]")
        for batch in progress_bar:
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items()}
            inputs['labels'] = inputs.pop('label')
            with autocast('cuda'):
                outputs = model(**inputs)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        val_loss = evaluate(model, val_loader, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            model.save_pretrained(model_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                break

def evaluate(model, dataloader, device):
    model.eval()
    loss_total = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = {k: v.to(device) for k, v in batch.items()}
            inputs['labels'] = inputs.pop('label')
            outputs = model(**inputs)
            loss_total += outputs.loss.item()
    avg_loss = loss_total / len(dataloader)
    return avg_loss

def load_model_and_tokenizer(model_path, tokenizer_path, num_labels=2):
    model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

if __name__ == "__main__":
    base_model_path = "C:/Ai/RoBERTa/bookcorpus_vocab"
    model, tokenizer = load_model_and_tokenizer(base_model_path, base_model_path)

    human_texts = extract_text_from_pdfs('C:/Ai/human-pdfs')
    ai_texts = extract_text_from_pdfs('C:/Ai/ai-pdfs')

    texts = human_texts + ai_texts
    labels = [0] * len(human_texts) + [1] * len(ai_texts)

    dataset = TextDataset(texts, labels, tokenizer, max_length=512)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

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
        patience=2
    )
