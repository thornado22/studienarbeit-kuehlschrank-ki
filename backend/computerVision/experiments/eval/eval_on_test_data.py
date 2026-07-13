import json
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torchvision import transforms

import os
import sys

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from computerVision.basemodels import load_config
from computerVision.train_model import build_model

THRESHOLD = 0.5


def load_test_labels(labels_path):
    with open(labels_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    labels_dict = {}
    for entry in data:
        img_name = entry["image"]
        if img_name.endswith(".jpg"):
            img_name = img_name[:-4]
        labels_dict[img_name] = entry["annotations"]
    return labels_dict


def predict_image(model, image_path, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    image = cv2.imread(image_path)
    if image is None:
        return None, None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    img_tensor = transform(pil_image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()
    preds = (probs >= THRESHOLD).astype(int)
    return preds, probs


def build_multi_class_confusion(y_true, y_pred, pred_idx, classes):
    """
    Build a class-vs-class confusion matrix for multi-label data.
    For each image, for each true class, find the top predicted class
    (among classes not already matched) and record a confusion.
    Returns a n_classes x n_classes matrix.
    """
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)

    for i in range(y_true.shape[0]):
        true_classes = np.where(y_true[i] == 1)[0]
        pred_classes = np.where(y_pred[i] == 1)[0]

        if len(true_classes) == 0:
            continue

        used_preds = set()
        for tc in true_classes:
            if tc in pred_classes:
                cm[tc, tc] += 1
                used_preds.add(tc)
            else:
                # Find best false prediction
                best_pred = -1
                best_score = -1
                for pc in pred_classes:
                    if pc not in used_preds and pc != tc:
                        cm[tc, pc] += 1
                        used_preds.add(pc)
                        break

        # Count extra predictions not matched to any true class
        for pc in pred_classes:
            if pc not in used_preds:
                # These are pure false positives, assign to a dummy row
                pass

    return cm


def evaluate(model_path, config_path, test_data_dir, output_dir=None):
    config = load_config(config_path)
    model_name = config.settings.model
    num_classes = len(config.meta.classes)
    classes = config.meta.classes

    print(f"Model: {model_name} | Classes: {num_classes}")
    print(f"Config: {config_path}")
    print(f"Test data: {test_data_dir}")
    print(f"Threshold: {THRESHOLD}\n")

    model = build_model(num_classes, model_name)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_labels = load_test_labels(os.path.join(test_data_dir, "labels.json"))
    test_images = [f for f in os.listdir(test_data_dir) if f.endswith(".jpg")]

    all_test_labels_set = set()
    for entry in test_labels.values():
        all_test_labels_set.update(entry)

    common_classes = [c for c in classes if c in all_test_labels_set]
    class_to_model_idx = {c: i for i, c in enumerate(classes)}
    class_idx = {c: i for i, c in enumerate(common_classes)}
    n = len(common_classes)

    if n == 0:
        print("No common classes found between model and test data annotations.")
        print("Model classes:", classes)
        print("Test annotations:", sorted(all_test_labels_set))
        return

    pred_idx = [class_to_model_idx[c] for c in common_classes]

    print(f"Common classes ({n}): {common_classes}")
    print(f"Running inference on {len(test_images)} images...\n")

    images_processed = 0
    y_true = np.zeros((len(test_images), n), dtype=int)
    y_pred = np.zeros((len(test_images), n), dtype=int)
    all_probs = np.zeros((len(test_images), num_classes), dtype=float)
    valid_mask = np.zeros(len(test_images), dtype=bool)

    for i, img_file in enumerate(test_images):
        img_name = img_file[:-4]
        if img_name not in test_labels:
            continue
        gt = test_labels[img_name]
        for j, cls in enumerate(common_classes):
            y_true[i, j] = 1 if cls in gt else 0
        pred, probs = predict_image(model, os.path.join(test_data_dir, img_file), device)
        if pred is not None:
            y_pred[i] = pred[pred_idx]
            all_probs[i] = probs[pred_idx]
            valid_mask[i] = True
            images_processed += 1

    if images_processed == 0:
        print("No images were successfully processed.")
        return

    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    all_probs = all_probs[valid_mask]

    print(f"Processed {images_processed}/{len(test_images)} images\n")

    # Per-class metrics
    print(f"{'Class':<25} {'P':>6} {'R':>6} {'F1':>6} {'Sup':>6} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}")
    print("-" * 85)
    f1s = []
    precisions = []
    recalls = []
    per_class_data = {}

    for i, cls in enumerate(common_classes):
        p = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        r = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        sup = y_true[:, i].sum()
        tp = int(((y_true[:, i] == 1) & (y_pred[:, i] == 1)).sum())
        fp = int(((y_true[:, i] == 0) & (y_pred[:, i] == 1)).sum())
        fn = int(((y_true[:, i] == 1) & (y_pred[:, i] == 0)).sum())
        tn = int(((y_true[:, i] == 0) & (y_pred[:, i] == 0)).sum())
        f1s.append(f1)
        precisions.append(p)
        recalls.append(r)
        per_class_data[cls] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "support": int(sup),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        }
        print(f"{cls:<25} {p:>6.3f} {r:>6.3f} {f1:>6.3f} {sup:>6d} {tp:>5d} {fp:>5d} {fn:>5d} {tn:>5d}")

    f1s = np.array(f1s)
    precisions = np.array(precisions)
    recalls = np.array(recalls)

    print(f"\n{'='*85}")
    print(f"Mean  P: {np.mean(precisions):.4f}  |  Mean  R: {np.mean(recalls):.4f}  |  Mean F1: {np.mean(f1s):.4f}")
    print(f"Median P: {np.median(precisions):.4f}  |  Median R: {np.median(recalls):.4f}  |  Median F1: {np.median(f1s):.4f}")

    sorted_idx = np.argsort(f1s)
    print(f"\nWorst 5: {', '.join(f'{common_classes[i]} (F1={f1s[i]:.3f})' for i in sorted_idx[:5])}")
    print(f"Best 5:  {', '.join(f'{common_classes[i]} (F1={f1s[i]:.3f})' for i in sorted_idx[-5:][::-1])}")

    # Build multi-class confusion: for each image, match true classes to predicted classes
    # and record which class was confused with which
    n_mc = len(common_classes)
    mc_confusion = np.zeros((n_mc, n_mc), dtype=int)

    for i in range(y_true.shape[0]):
        true_classes = np.where(y_true[i] == 1)[0]
        pred_classes = np.where(y_pred[i] == 1)[0]

        if len(true_classes) == 0:
            continue

        used = set()
        for tc in true_classes:
            if tc in pred_classes:
                mc_confusion[tc, tc] += 1
                used.add(tc)
            else:
                best_idx = -1
                best_prob = -1
                for pc in pred_classes:
                    if pc not in used and pc != tc:
                        if all_probs[i, pc] > best_prob:
                            best_prob = all_probs[i, pc]
                            best_idx = pc
                if best_idx >= 0:
                    mc_confusion[tc, best_idx] += 1
                    used.add(best_idx)

        for pc in pred_classes:
            if pc not in used:
                pass

    # Normalize confusion matrix by row for better visualization
    row_sums = mc_confusion.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    mc_confusion_norm = mc_confusion / row_sums

    # Save per-class confusion JSON
    mc_data = {
        "threshold": THRESHOLD,
        "num_images": int(y_true.shape[0]),
        "classes": common_classes,
        "per_class": per_class_data
    }

    json_path = os.path.join(output_dir, "per_class_confusion.json") if output_dir else "per_class_confusion.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(mc_data, f, indent=2, ensure_ascii=False)
    print(f"\nPer-class confusion JSON saved: {json_path}")

    # Plot 1: Binary confusion matrix
    cm_binary = confusion_matrix(y_true.flatten(), y_pred.flatten(), labels=[1, 0])
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_binary, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["Positive", "Negative"],
           yticklabels=["Positive", "Negative"],
           xlabel="Predicted", ylabel="Actual",
           title=f"Binary Confusion Matrix (Threshold={THRESHOLD})")
    fmt = "d"
    thresh_val = cm_binary.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm_binary[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm_binary[i, j] > thresh_val else "black")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "confusion_matrix.png") if output_dir else "confusion_matrix.png"
    fig.savefig(out_path, dpi=150)
    print(f"Confusion matrix saved: {out_path}")
    plt.close()

    # Plot 2: Multi-class confusion matrix (where mixups happen)
    fig, ax = plt.subplots(figsize=(max(10, n_mc * 0.6), max(8, n_mc * 0.5)))
    im2 = ax.imshow(mc_confusion_norm, interpolation="nearest", cmap=plt.cm.RdYlGn)
    ax.figure.colorbar(im2, ax=ax)
    ax.set(xticks=range(n_mc), yticks=range(n_mc),
           xticklabels=common_classes, yticklabels=common_classes,
           xlabel="Predicted Class", ylabel="True Class",
           title=f"Class Confusion Matrix (Threshold={THRESHOLD})\nRows=True, Cols=Predicted (normalized by row)")
    fmt2 = ".2f"
    for i in range(n_mc):
        for j in range(n_mc):
            val = mc_confusion_norm[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)
            if mc_confusion[i, j] > 0:
                ax.text(j, i, f"\n(n={mc_confusion[i, j]})", ha="center", va="top", fontsize=6, color="gray")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "class_confusion_matrix.png") if output_dir else "class_confusion_matrix.png"
    fig.savefig(out_path, dpi=150)
    print(f"Class confusion matrix saved: {out_path}")
    plt.close()

    # Plot 3: F1 scores bar chart
    sorted_by_f1 = sorted(enumerate(f1s), key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(max(10, n_mc * 0.6), 6))
    names = [common_classes[i] for i, _ in sorted_by_f1]
    vals = [v for _, v in sorted_by_f1]
    colors = ["green" if v >= 0.7 else "orange" if v >= 0.4 else "red" for v in vals]
    ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("F1 Score")
    ax.set_title(f"Per-Class F1 Scores (Threshold={THRESHOLD})")
    ax.set_xlim(0, 1)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "f1_scores.png") if output_dir else "f1_scores.png"
    fig.savefig(out_path, dpi=150)
    print(f"F1 scores chart saved: {out_path}")
    plt.close()

    print("\nDone!")


if __name__ == "__main__":
    model = "test_sahne"
    ROOT = "computerVision"
    evaluate(
        model_path=os.path.join(ROOT, "models", f"{model}.pth"),
        config_path=os.path.join(ROOT, "models", f"{model}.json"),
        test_data_dir=os.path.join(ROOT, "images/test_images"),
        output_dir=os.path.join(ROOT, "eval", model),
    )
