import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import os

from dataset import SimCSEDataset
from model import SimCSEModel
from loss import SimCSELoss


# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"   # good for CPU
CSV_PATH = "data/sentences_2019.csv"
BATCH_SIZE = 16          # keep small on CPU
MAX_LENGTH = 64
EPOCHS = 1
LR = 3e-5
TEMPERATURE = 0.05
DEVICE = torch.device("cpu")


def main():
    print(f"Using device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")

    # Dataset
    train_dataset = SimCSEDataset(
        csv_path=CSV_PATH,
        split="train",
        model_name=MODEL_NAME,
        max_length=MAX_LENGTH
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    # Model + Loss + Optimizer
    model = SimCSEModel(model_name=MODEL_NAME, temperature=TEMPERATURE).to(DEVICE)
    criterion = SimCSELoss(temperature=TEMPERATURE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in progress:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            # Two forward passes → two different dropout views
            z1, _ = model(input_ids, attention_mask)
            z2, _ = model(input_ids, attention_mask)

            loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/simcse_minilm.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()