import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from models.rtlr_model import CTCRecognitionModel
from utils.label_converter import LabelConverter

def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    label_tensors = [torch.tensor(l, dtype=torch.long) for l in labels]
    labels_padded = pad_sequence(label_tensors, batch_first=True, padding_value=0)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    return images, labels_padded, label_lengths

def train(model, dataloader, converter, device, epochs=10):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CTCLoss(blank=converter.blank_idx, zero_infinity=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for images, labels_padded, label_lengths in dataloader:
            images = images.to(device)
            labels_padded = labels_padded.to(device)
            label_lengths = label_lengths.to(device)

            logits = model(images)
            log_probs = logits.log_softmax(2).permute(1, 0, 2)  # (T, N, C)
            input_lengths = torch.full(
                size=(logits.size(0),), fill_value=logits.size(1), dtype=torch.long
            ).to(device)

            loss = criterion(log_probs, labels_padded, input_lengths, label_lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")
