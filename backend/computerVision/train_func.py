from torch.utils.data import Dataset, WeightedRandomSampler
import torch
import numpy as np
import os
import cv2
torch.backends.cudnn.benchmark = True
from computerVision.basemodels import TrainingResults


class MultiLabelDataset(Dataset):
    """
    Loads a multilabel Dataset
    """
    def __init__(self, df, images_dir, transform_builder):
        self.df = df
        self.images_dir = images_dir
        self.transforms = transform_builder.pipeline
        self.filenames = df.index.to_list()
        self.label_cols = df.columns

    def __len__(self):
        return len(self.filenames) 
    
    def __getitem__(self, idx):
        name = self.filenames[idx]
        fname = name + ".jpg"
        path = os.path.join(self.images_dir, fname)

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transforms(image=image)["image"]

        labels = torch.tensor(
            self.df.loc[name].values,
            dtype=torch.float32
        )

        return image, labels

class BinaryDataset(Dataset):
    def __init__(self, df, images_dir, transform_builder):
        self.df = df
        self.images_dir = images_dir
        self.transforms = transform_builder.pipeline
        self.filenames = df.index.to_list()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        path = os.path.join(self.images_dir, name + ".jpg")

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)["image"]

        label = torch.tensor(
            [self.df.loc[name, "target"]],
            dtype=torch.float32
        )

        return image, label


class EarlyStopping:
    """
    Stop training if validation f1 does not improve after `patience` amount of epochs
    """
    
    def __init__(self, patience=10, min_delta=0.0, save_path="best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path

        self.best_f1 = (-float("inf"), [])
        self.train_loss = 0
        self.val_loss = 0
        self.epoch = 0
        self.counter = 0
        self.early_stop = False

    def __call__(self, f1, train_loss, val_loss, epoch, model):
        # improvement
        if f1[0] > self.best_f1[0] + self.min_delta:
            self.best_f1 = f1
            self.train_loss = train_loss
            self.val_loss = val_loss
            self.epoch = epoch
            self.counter = 0

            # save best model
            torch.save(model.state_dict(), self.save_path)

            print(f"Validation F1 improved to {f1[0]:.4f}. Model saved.")

        else:
            self.counter += 1
            print(f"No improvement for {self.counter}/{self.patience} epochs")

            if self.counter >= self.patience:
                self.early_stop = True
    
    def get_infos(self):
        return TrainingResults(
            val_loss = self.val_loss,
            train_loss = self.train_loss,
            val_class_f1 = self.best_f1[1],
            val_f1 = self.best_f1[0],
            stop_epcoh = self.epoch)

def compute_pos_weights(df):
    """
    Used to reduce effect of class imbalance by finding amounts of positive samples per class
    """
    
    # number of samples
    N = len(df)

    # positives per class
    pos_counts = df.sum(axis=0).values
    neg_counts = N - pos_counts

    # avoid division by zero
    pos_counts = torch.tensor(pos_counts, dtype=torch.float32)
    neg_counts = torch.tensor(neg_counts, dtype=torch.float32)

    pos_weight = neg_counts / (pos_counts + 1e-6)

    return pos_weight



def create_sampler(df):
    # weight per sample = sum of inverse class frequency
    class_counts = df.sum(axis=0)
    class_weights = 1.0 / (class_counts + 1e-6)

    sample_weights = df.values @ class_weights.values
    sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = torch.sigmoid(outputs)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    return total_loss / len(loader), all_preds, all_labels

def compute_f1(preds, labels, threshold=0.5):
    preds_bin = (preds > threshold).float()

    tp = (preds_bin * labels).sum(dim=0)
    fp = (preds_bin * (1 - labels)).sum(dim=0)
    fn = ((1 - preds_bin) * labels).sum(dim=0)

    f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)

    return f1.mean().item()

def compute_f1_per_class(preds, labels, thresholds):
    preds_bin = (preds > thresholds).float()

    tp = (preds_bin * labels).sum(dim=0)
    fp = (preds_bin * (1 - labels)).sum(dim=0)
    fn = ((1 - preds_bin) * labels).sum(dim=0)

    f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
    return f1.mean().item(), f1.tolist()


def optimize_thresholds(preds, labels):
    preds = preds.numpy()
    labels = labels.numpy()

    thresholds = np.linspace(0, 1, 100)

    best_thresholds = []

    for c in range(preds.shape[1]):
        p = preds[:, c]
        y = labels[:, c]

        f1_scores = []

        for t in thresholds:
            pred_bin = (p >= t)

            tp = (pred_bin & y.astype(bool)).sum()
            fp = (pred_bin & ~y.astype(bool)).sum()
            fn = (~pred_bin & y.astype(bool)).sum()

            f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
            f1_scores.append(f1)

        best_thresholds.append(thresholds[np.argmax(f1_scores)])

    return np.array(best_thresholds)