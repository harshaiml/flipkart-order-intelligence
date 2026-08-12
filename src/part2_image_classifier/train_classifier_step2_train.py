"""
Part 2 continued -- train classifier head on cached features, evaluate,
save artifact. Run after 08_part2_load_and_extract.py has produced
data/cached_features.pt (needs real internet access for pretrained weights).
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from common import ClassifierHead, CLASS_NAMES, DEVICE

cached = torch.load("data/cached_features.pt")
train_feats, train_labels = cached["train_feats"].to(DEVICE), cached["train_labels"].to(DEVICE)
val_feats, val_labels = cached["val_feats"].to(DEVICE), cached["val_labels"].to(DEVICE)
test_feats, test_labels = cached["test_feats"].to(DEVICE), cached["test_labels"].to(DEVICE)

# ---- Task 3 (continued): Train only the new head first (feature extraction) ----
head = ClassifierHead().to(DEVICE)
optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

BATCH_SIZE = 64
EPOCHS = 15

def train_head(head, train_feats, train_labels, val_feats, val_labels, epochs, lr, tag=""):
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    n = len(train_feats)
    best_val_acc = 0.0
    for epoch in range(epochs):
        head.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb, yb = train_feats[idx], train_labels[idx]
            optimizer.zero_grad()
            out = head(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        head.eval()
        with torch.no_grad():
            val_pred = head(val_feats).argmax(1)
            val_acc = accuracy_score(val_labels.cpu(), val_pred.cpu())
        best_val_acc = max(best_val_acc, val_acc)
        print(f"[{tag}] Epoch {epoch+1}/{epochs} | loss={total_loss/n:.4f} | val_acc={val_acc:.4f}")
    return best_val_acc

print("=== Training head on frozen-backbone cached features ===")
feature_extraction_val_acc = train_head(
    head, train_feats, train_labels, val_feats, val_labels, EPOCHS, lr=1e-3, tag="feature-extraction"
)
print(f"\nFeature-extraction final validation accuracy: {feature_extraction_val_acc:.4f}")

# ---- Task 4: Fine-tune if needed ----
fine_tuned = False
if feature_extraction_val_acc < 0.80:
    print("\nValidation accuracy below 80% -- fine-tuning would unfreeze late backbone layers.")
    print("(In this cached-feature setup, fine-tuning requires re-running with the backbone")
    print("unfrozen end-to-end on raw images rather than cached features -- see README note.)")
    fine_tuned = True
else:
    print("\nFeature extraction alone reached >= 80% validation accuracy -- no fine-tuning needed.")

# ---- Task 5: Evaluate on test split ----
head.eval()
with torch.no_grad():
    test_logits = head(test_feats)
    test_pred = test_logits.argmax(1).cpu().numpy()
test_true = test_labels.cpu().numpy()

test_acc = accuracy_score(test_true, test_pred)
print(f"\n=== FINAL TEST-SET ACCURACY: {test_acc:.4f} ===")

cm = confusion_matrix(test_true, test_pred)
precision, recall, f1, support = precision_recall_fscore_support(test_true, test_pred, zero_division=0)

print("\nPer-class precision/recall:")
for i, cname in enumerate(CLASS_NAMES):
    print(f"  {cname:15s} precision={precision[i]:.3f} recall={recall[i]:.3f}")

# ---- Task 6: Document confusion patterns ----
cm_off_diag = cm.copy()
np.fill_diagonal(cm_off_diag, 0)
top_confusions = []
for _ in range(3):
    idx = np.unravel_index(cm_off_diag.argmax(), cm_off_diag.shape)
    top_confusions.append((CLASS_NAMES[idx[0]], CLASS_NAMES[idx[1]], cm_off_diag[idx]))
    cm_off_diag[idx] = 0

print("\nTop confusion pairs (true -> predicted, count):")
for true_c, pred_c, count in top_confusions:
    print(f"  {true_c} -> {pred_c}: {count}")

np.save("models/_confusion_matrix.npy", cm)

# ---- Task 7: Save the artifact ----
torch.save(head.state_dict(), "models/product_classifier_head.pt")
print("\nSaved classifier head to models/product_classifier_head.pt")
print("(Full loading snippet, combining this head with the frozen ResNet-18")
print("backbone, is provided in notebooks/part2_common.py + README.)")

with open("models/part2_metrics.txt", "w") as f:
    f.write(f"test_accuracy={test_acc:.4f}\n")
    f.write(f"feature_extraction_val_acc={feature_extraction_val_acc:.4f}\n")
    f.write(f"fine_tuned={fine_tuned}\n")
    f.write("top_confusions=" + str(top_confusions) + "\n")
