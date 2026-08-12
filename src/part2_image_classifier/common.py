"""
Shared utilities for Part 2 (and consumed by Part 3's classify_product_image tool).
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class ClassifierHead(nn.Module):
    """New classifier head sized for 10 output classes, trained on top of
    frozen ResNet-18 512-d features."""
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


def build_backbone():
    """Frozen, ImageNet-pretrained ResNet-18 with its final FC layer removed,
    so it outputs 512-d feature vectors instead of 1000-class ImageNet logits."""
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()
    return backbone.to(DEVICE)


_backbone_singleton = None
_head_singleton = None


def _load_models(head_path="models/product_classifier_head.pt"):
    global _backbone_singleton, _head_singleton
    if _backbone_singleton is None:
        _backbone_singleton = build_backbone()
    if _head_singleton is None:
        _head_singleton = ClassifierHead().to(DEVICE)
        _head_singleton.load_state_dict(torch.load(head_path, map_location=DEVICE))
        _head_singleton.eval()
    return _backbone_singleton, _head_singleton


def classify_product_image(image_path: str) -> dict:
    """Part 3's tool: loads Part 2's saved classifier and returns the
    predicted category label plus the model's confidence for that prediction.
    Run against real .png files exported to data/sample_images/.
    """
    backbone, head = _load_models()
    img = Image.open(image_path).convert("L")  # ensure single-channel input
    x = INFERENCE_TRANSFORM(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats = backbone(x)
        logits = head(feats)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])
    return {
        "predicted_category": CLASS_NAMES[pred_idx],
        "confidence": round(confidence, 4),
    }
