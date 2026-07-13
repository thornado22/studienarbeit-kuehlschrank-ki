import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
import numpy as np


def multilabel_confusion_matrix(y_true, y_pred, threshold=0.5, num_classes=10):
    y_pred = (y_pred >= threshold).int()

    cm = torch.zeros((num_classes, 4))  
    # columns: TP, FP, TN, FN

    for c in range(num_classes):
        true_c = y_true[:, c]
        pred_c = y_pred[:, c]

        tp = ((pred_c == 1) & (true_c == 1)).sum().item()
        fp = ((pred_c == 1) & (true_c == 0)).sum().item()
        tn = ((pred_c == 0) & (true_c == 0)).sum().item()
        fn = ((pred_c == 0) & (true_c == 1)).sum().item()

        cm[c] = torch.tensor([tp, fp, tn, fn])

    return cm

def cooccurrence_matrix(y_true, y_pred, label_names, threshold=0.5):
    y_pred = (y_pred >= threshold).int()

    num_classes = y_true.shape[1]
    matrix = torch.zeros((num_classes, num_classes))

    for i in range(num_classes):
        for j in range(num_classes):
            matrix[i, j] = ((y_true[:, i] == 1) & (y_pred[:, j] == 1)).sum()

    plt.figure(figsize=(12, 10))

    sns.heatmap(
            matrix.numpy(),
            xticklabels=label_names,
            yticklabels=label_names,
            cmap="Blues"
        )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Multi-label Co-occurrence Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return matrix