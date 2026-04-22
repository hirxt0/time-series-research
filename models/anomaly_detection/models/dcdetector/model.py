import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, win_size = x.shape
        x = x.view(batch, -1, self.patch_size)
        return self.projection(x)

class DualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn1 = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attn2 = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> tuple:
        _, w1 = self.attn1(x, x, x)
        _, w2 = self.attn2(x, x, x)
        return w1, w2

class DCDetectorLite(nn.Module):
    def __init__(self, patch_size: int, d_model: int = 64, n_heads: int = 4):
        super().__init__()
        self.embedding = PatchEmbedding(patch_size, d_model)
        self.attention = DualAttentionBlock(d_model, n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        w1, w2 = self.attention(x)
        return torch.mean(torch.abs(w1 - w2), dim=(1, 2))

def train_dc(df, win_size=120, patch_size=6, epochs=1, batch_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data = df['value'].values.astype(np.float32)
    mean, std = data.mean(), data.std()
    norm_data = (data - mean) / (std + 1e-6)

    step = win_size // 2
    windows = [norm_data[i:i+win_size] for i in range(0, len(norm_data)-win_size, step)]
    x_tensor = torch.tensor(np.array(windows), dtype=torch.float32)
    
    model = DCDetectorLite(patch_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(x_tensor), batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for batch in loader:
            optimizer.zero_grad()
            loss = model(batch[0].to(device)).mean()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} complete")

    model.eval()
    with torch.no_grad():
        all_scores = []
        test_loader = DataLoader(TensorDataset(x_tensor), batch_size=batch_size, shuffle=False)
        for batch in test_loader:
            s = model(batch[0].to(device)).cpu().numpy()
            all_scores.extend(s)
        scores = np.array(all_scores)

    full_scores = np.zeros(len(norm_data))
    counts = np.zeros(len(norm_data))
    for i, idx in enumerate(range(0, len(norm_data)-win_size, step)):
        full_scores[idx:idx+win_size] += scores[i]
        counts[idx:idx+win_size] += 1
    
    final_scores = full_scores / np.maximum(counts, 1)
    
    df['score'] = final_scores
    df['anomaly'] = (final_scores > np.percentile(final_scores, 98)).astype(int)
    return df