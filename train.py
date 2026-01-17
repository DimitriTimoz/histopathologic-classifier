import os
import time

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from typing import List
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict, Counter
from typing import Tuple, Optional, Sequence
import torchvision.transforms as T
from torch.utils.data import Subset, WeightedRandomSampler
from dataset import FilenameLabelImageDataset
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import tqdm
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report
from model import CNN_PDA, CNN
import pandas as pd

DATA_FOLDER = 'mask_output/images'

def make_transforms_old(image_size: int = 224, train: bool = True):
    # Minimal baseline; swap/extend for augmentation later.
    ops = [
        T.Resize((image_size, image_size)),
    ]
    if train:
        # Example augmentation hook (keep minimal):
        ops.append(T.RandomHorizontalFlip(p=0.5))
        ops.append(T.RandomVerticalFlip(p=0.5))
        ops.append(T.RandomRotation(degrees=90))

    ops += [
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
    return T.Compose(ops)

def make_transforms(train=True):
    ops = []
    if train:
        ops.append(T.RandomCrop(224))
        ops += [
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
        ]
    #else:
    #    ops.append(T.RandomCrop(224))

    ops += [
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ]
    return T.Compose(ops)

def stratified_split_indices(labels: Sequence[int], val_frac: float = 0.15, test_frac: float = 0.15) -> Tuple[list[int], list[int], list[int]]:
    by_class = defaultdict(list)
    for i, y in enumerate(labels):
        by_class[int(y)].append(i)

    train_idx  = []
    val_idx = []
    test_idx = []

    for y, idxs in by_class.items():
        idxs = idxs.copy()
        perm = torch.randperm(len(idxs)).tolist()
        idxs = [idxs[i] for i in perm]
        n = len(idxs)
        n_test = int(round(test_frac * n))
        n_val = int(round(val_frac * n))
        if n >= 3:
            n_test = max(1, n_test)
            n_val = max(1, n_val)
        if n_test + n_val >= n:
            n_test = min(n_test, max(0, n - 1))
            n_val = min(n_val, max(0, n - 1 - n_test))
        n_train = n - n_val - n_test
        if n_train <= 0 and n > 0:
            n_train = 1
            if n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1
        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train:n_train + n_val])
        test_idx.extend(idxs[n_train + n_val:])

    def _shuffle(xs: list[int]) -> list[int]:
        if not xs:
            return xs
        perm = torch.randperm(len(xs)).tolist()
        return [xs[i] for i in perm]

    return _shuffle(train_idx), _shuffle(val_idx), _shuffle(test_idx)

def make_class_weights(targets: Sequence[int], num_classes: int) -> torch.Tensor:
    counts = torch.bincount(torch.as_tensor(list(targets), dtype=torch.long), minlength=num_classes).clamp_min(1)
    total = counts.sum().float()
    weights = total / (num_classes * counts.float())
    return weights

def make_dataloaders(
    data_folder: str,
    batch_size: int = 32,
    image_size: int = 224,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
    exclude_classes: Optional[set[str]] = None,
    balance_train: bool = True,
 ):
    #train_tf = make_transforms(image_size=image_size, train=True)
    #eval_tf = make_transforms(image_size=image_size, train=False)
    train_tf = make_transforms(train=True)
    eval_tf = make_transforms(train=False)

    base_ds = FilenameLabelImageDataset(data_folder, transform=None, exclude_classes=exclude_classes)
    train_idx, val_idx, test_idx = stratified_split_indices(base_ds.targets, val_frac, test_frac)

    train_ds = base_ds.with_transform(train_tf)
    val_ds = base_ds.with_transform(eval_tf)
    test_ds = base_ds.with_transform(eval_tf)

    train_subset = Subset(train_ds, train_idx)
    val_subset = Subset(val_ds, val_idx)
    test_subset = Subset(test_ds, test_idx)

    num_classes = len(base_ds.class_to_idx)
    train_targets = [base_ds.targets[i] for i in train_idx]
    class_weights = make_class_weights(train_targets, num_classes=num_classes)

    sampler = None
    if balance_train:
        sample_weights = class_weights[torch.as_tensor(train_targets, dtype=torch.long)]
        rng = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_targets), replacement=True, generator=rng)

    pin_memory = torch.cuda.is_available()
    loader_kwargs = {"num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        # Keeps workers alive across epochs and prefetches batches to hide disk latency.
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})

    loaders = {
        "train": DataLoader(train_subset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler, **loader_kwargs),
        "val": DataLoader(val_subset, batch_size=1, shuffle=False, **loader_kwargs),
        "test": DataLoader(test_subset, batch_size=1, shuffle=False, **loader_kwargs),
    }
    # J'ai mis batch_size=1 pour val et test
    meta = {
        "class_to_idx": base_ds.class_to_idx,
        "idx_to_class": base_ds.idx_to_class,
        "counts": base_ds.class_counts(),
        "sizes": {"train": len(train_subset), "val": len(val_subset), "test": len(test_subset)},
        "class_weights": class_weights,
        "targets": base_ds.targets,
        "split_indices": {"train": train_idx, "val": val_idx, "test": test_idx},
    }
    return loaders, meta

CFG = {
    "data_folder": DATA_FOLDER,
    "batch_size": 16,
    "image_size": 224,
    "splits": {"val": 0.15, "test": 0.15},
    "num_workers": 0,
    "exclude_classes": {"Region"},
    # Handle class imbalance
    "balance_train": True,   
    "weighted_loss": True, 
}

dataloaders, data_meta = make_dataloaders(
    data_folder=CFG["data_folder"],
    batch_size=CFG["batch_size"],
    image_size=CFG["image_size"],
    val_frac=CFG["splits"]["val"],
    test_frac=CFG["splits"]["test"],
    num_workers=CFG["num_workers"],
    exclude_classes=CFG["exclude_classes"],
    balance_train=CFG["balance_train"],
 )

print("Classes:", data_meta["class_to_idx"])
print("Counts:", data_meta["counts"])
print("Split sizes:", data_meta["sizes"])
print("Class weights:", data_meta["class_weights"].tolist())

labels = list(data_meta["idx_to_class"].values())
targets = data_meta["targets"]
split_indices = data_meta["split_indices"]

to_pil = ToPILImage()
train_subset = dataloaders['train'].dataset
loader_visu = DataLoader(train_subset, batch_size=4, shuffle=True, num_workers=0)
class_names = list(data_meta["idx_to_class"].values())

mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def denormalize(img):
    return img * std + mean

gamma = 0.80
model = CNN_PDA(num_classes=len(data_meta["class_to_idx"]), gamma=gamma)
#model = CNN(num_classes=len(data_meta["class_to_idx"]))
model_dir = f"model_{time.strftime('%Y%m%d-%H:%M')}"
os.mkdir(model_dir)
print(f"Résultats seront sauvegardés dans {model_dir}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

loss_weight = None
if CFG.get("weighted_loss", False):
    loss_weight = data_meta["class_weights"].to(device)
criterion = nn.CrossEntropyLoss(weight=loss_weight)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Metrics
history = [] 
epoch_train_loss = []
epoch_train_acc = []
epoch_val_loss = []
history_val = []
best_val_acc = 0.0
NUM_EPOCHS = 20
for epoch in tqdm.tqdm(range(NUM_EPOCHS)):
    model.train()
    running_loss = 0.0
    running_samples = 0
    running_correct = 0
    use_PDA = epoch > 2
    for x, y in dataloaders["train"]:
        x, y = x.to(device), y.to(device)
        outputs = model(x, apply_attention=use_PDA)
        #outputs = model(x)
        loss = criterion(outputs, y)
        history.append(loss.item())
        running_loss += loss.item() * y.size(0)
        running_samples += y.size(0)
        _, preds = torch.max(outputs, 1)
        running_correct += (preds == y).sum().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # epoch stats on training set
    train_loss_epoch = running_loss / running_samples if running_samples > 0 else 0.0
    train_acc_epoch = running_correct / running_samples if running_samples > 0 else 0.0
    epoch_train_loss.append(train_loss_epoch)
    epoch_train_acc.append(train_acc_epoch)

    # Validation: compute loss and accuracy
    model.eval()
    val_loss_accum = 0.0
    val_samples = 0
    val_correct = 0
    with torch.no_grad():
        for x, y in dataloaders['val']:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            val_loss_accum += loss.item() * y.size(0)
            val_samples += y.size(0)
            _, preds = torch.max(logits, 1)
            val_correct += (preds == y).sum().item()
        
    val_loss_epoch = val_loss_accum / val_samples if val_samples > 0 else 0.0
    val_acc_epoch = val_correct / val_samples if val_samples > 0 else 0.0
    if val_acc_epoch > best_val_acc:
        best_val_acc = val_acc_epoch
        torch.save(model.state_dict(), f"{model_dir}/best_model.pth")
    epoch_val_loss.append(val_loss_epoch)
    history_val.append(val_acc_epoch)

    print(f"Epoch {epoch+1}: train_loss={train_loss_epoch:.4f}, train_acc={train_acc_epoch:.4f}, val_loss={val_loss_epoch:.4f}, val_acc={val_acc_epoch:.4f}")

# After training: compute test metrics and confusion matrix
model.eval()
y_true = []
y_pred = []
y_prob = []
with torch.no_grad():
    for x, y in dataloaders['test']:
        x = x.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy().tolist()
        y_pred.extend(preds)
        y_true.extend(y.numpy().tolist())
        y_prob.extend(probs.cpu().numpy().tolist())

labels = [data_meta['idx_to_class'][i] for i in range(len(data_meta['idx_to_class']))]
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(labels))), zero_division=0)

print('\nClassification report:')
print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))


# Epoch-level losses and accuracies
epochs = list(range(1, len(epoch_train_loss) + 1))
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, epoch_train_loss, '-o', label='Train loss')
plt.plot(epochs, epoch_val_loss, '-o', label='Val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, epoch_train_acc, '-o', label='Train acc')
plt.plot(epochs, history_val, '-o', label='Val acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy per epoch')
plt.legend()
plt.savefig(f"{model_dir}/training_plots.png")

# Confusion matrix heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Test set)')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.savefig(f"{model_dir}/confusion_matrix.png")

