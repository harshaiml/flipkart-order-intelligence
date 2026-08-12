"""
Part 2 -- Product Image Categoriser via Transfer Learning
Run this on a machine/Colab with normal internet access (needs to download
pretrained ImageNet weights from download.pytorch.org, which is blocked in
this sandbox but works everywhere else, including Google Colab free tier).
"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ---- Task 1: Load dataset with stratified validation split ----
# torchvision.datasets.FashionMNIST(root=..., download=True) pulls from the
# canonical Zalando Research source (same data we already verified in sandbox).
raw_train = datasets.FashionMNIST(root="./data", train=True, download=True)
raw_test = datasets.FashionMNIST(root="./data", train=False, download=True)

# Stratified validation split (at least 5000 images) out of the training set
train_targets = np.array(raw_train.targets)
train_idx, val_idx = train_test_split(
    np.arange(len(raw_train)), test_size=5000, stratify=train_targets, random_state=42
)
print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(raw_test)}")

# ---- Task 2: Preprocess for a pretrained backbone ----
# Replicate grayscale -> 3 channels, resize to 224x224 (ResNet-18 input size),
# normalize with ImageNet mean/std (the stats ResNet-18 was originally trained with).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

full_train = datasets.FashionMNIST(root="./data", train=True, download=False, transform=transform)
full_test = datasets.FashionMNIST(root="./data", train=False, download=False, transform=transform)

train_set = Subset(full_train, train_idx)
val_set = Subset(full_train, val_idx)

# ---- Task 3: Build the transfer-learning model ----
# Speed tip from the brief: since the backbone is frozen, extract+cache its
# output features ONCE, then train only the small head on cached features.
# This turns an hours-long CPU loop into minutes.

def build_backbone():
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()  # remove final classification layer -> outputs 512-d features
    for param in backbone.parameters():
        param.requires_grad = False  # freeze early/middle (and here, all) backbone layers
    backbone.eval()
    return backbone.to(DEVICE)


@torch.no_grad()
def extract_features(dataset, backbone, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    feats, labels = [], []
    for imgs, y in loader:
        imgs = imgs.to(DEVICE)
        f = backbone(imgs)
        feats.append(f.cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


class ClassifierHead(nn.Module):
    """New classifier head sized for 10 output classes."""
    def __init__(self, in_features=512, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    print("Building frozen backbone (ResNet-18, ImageNet-pretrained)...")
    backbone = build_backbone()

    print("Extracting cached features for train/val/test (one pass each)...")
    train_feats, train_labels = extract_features(train_set, backbone)
    val_feats, val_labels = extract_features(val_set, backbone)
    test_feats, test_labels = extract_features(full_test, backbone)

    torch.save({
        "train_feats": train_feats, "train_labels": train_labels,
        "val_feats": val_feats, "val_labels": val_labels,
        "test_feats": test_feats, "test_labels": test_labels,
    }, "data/cached_features.pt")
    print("Cached features saved to data/cached_features.pt")
    print(f"Batch size used: 64, Optimizer: Adam (next step), Backbone: ResNet-18 frozen")
