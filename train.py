import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

DATA_ROOT      = "data"             
CLASS_MAP_FILE = "class_map.json"   
CHECKPOINT_DIR = "checkpoints"      
BATCH_SIZE     = 32
NUM_EPOCHS     = 10
LR_HEAD        = 1e-3
WEIGHT_DECAY   = 1e-4
PATIENCE       = 2           
DEVICE         = torch.device("cpu") 

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

with open(CLASS_MAP_FILE, "r") as f:
    id2label = json.load(f)
num_classes = len(id2label)

def main():
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dir = os.path.join(DATA_ROOT, "train")
    val_dir   = os.path.join(DATA_ROOT, "val")
    test_dir  = os.path.join(DATA_ROOT, "test")

    if not (os.path.isdir(train_dir) and os.path.isdir(val_dir) and os.path.isdir(test_dir)):
        raise RuntimeError(
            "Expected data/train, data/val, data/test each with subfolders for classes."
        )

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset   = datasets.ImageFolder(val_dir,   transform=val_transforms)
    test_dataset  = datasets.ImageFolder(test_dir,  transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=False)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=False)

    model = models.efficientnet_b0(pretrained=True)

    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=1,
        factor=0.5
    )

    best_val_loss      = float("inf")
    epochs_no_improve  = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        correct  = 0
        total    = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Val]"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss    = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc      = correct / total

        print(
            f"\n→ Epoch {epoch}: "
            f"Train Loss={avg_train_loss:.4f}, "
            f"Val Loss={avg_val_loss:.4f}, "
            f"Val Acc={val_acc:.4f}\n"
        )

        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"efficientnet_b0_epoch{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved new best model → {ckpt_path}\n")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"No improvement for {PATIENCE} consecutive epochs → Early stopping.\n")
                break

    ckpt_files = sorted(os.listdir(CHECKPOINT_DIR))
    best_ckpt  = ckpt_files[-1]
    print(f"Loading best checkpoint: {best_ckpt}")

    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, best_ckpt)))
    model.eval()

    correct = 0
    total   = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="[Test Evaluation]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    test_acc = correct / total
    print(f"\nFinal Test Accuracy = {test_acc:.4f}\n")


if __name__ == "__main__":
    main()
