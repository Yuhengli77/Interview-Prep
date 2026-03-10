import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 1. fake dataset
X = torch.randn(1000, 20)
y = torch.randint(0, 3, (1000,))   # 3-class classification

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 2. simple model
model = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Linear(64, 3)
)

# 3. loss + optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 4. training loop
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_x, batch_y in dataloader:
        # forward
        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")


# validation set
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch_x, batch_y in dataloader:
        logits = model(batch_x)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

acc = correct / total
print("Accuracy:", acc)