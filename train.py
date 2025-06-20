import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from models.densenet import get_densenet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

data_dir = "data/chest_xray"

train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_transform)
val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=val_transform)

# Debug prints for dataset loading
print(f"✅ Loaded train dataset with {len(train_dataset)} images")
print(f"✅ Loaded val dataset with {len(val_dataset)} images")

# Use num_workers=0 for Windows to avoid multiprocessing issues
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

model = get_densenet(num_classes=2, pretrained=True)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train_one_epoch():
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate():
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# Wrap training loop to support Windows multiprocessing
if __name__ == "__main__":
    print("🚀 Starting training...")

    num_epochs = 2
    best_val_acc = 0

    for epoch in range(num_epochs):
        print(f"\n📘 Epoch {epoch + 1}/{num_epochs}")
        train_loss, train_acc = train_one_epoch()
        val_loss, val_acc = evaluate()

        print(f"✅ Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"✅ Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("💾 Best model saved!")
