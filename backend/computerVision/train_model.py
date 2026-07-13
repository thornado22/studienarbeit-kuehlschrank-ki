from torch.utils.data import DataLoader
import torch

import os
import sys

backend_dir = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, backend_dir)

from computerVision.feature_eng import AlbumentationsBuilder
from computerVision.loader import LabelLoader
import torch.nn as nn
from torchvision.models import efficientnet_b0, resnet18, resnet50, convnext_tiny, mobilenet_v3_small
from sklearn.model_selection import train_test_split
torch.backends.cudnn.benchmark = True
import computerVision.train_func as t
from computerVision.basemodels import Config, TrainingPaths, TrainingSettings, Metadata, load_config, save_results

    

def build_model(num_classes, model):
    if model == "efficientnet_b0":
        model = efficientnet_b0(weights="DEFAULT")
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(
            in_features,
            num_classes
        )
        return model
    elif model == "resnet18":
        model = resnet18(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model == "resnet50":
        model = resnet50(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model == "convnext_tiny":
        model = convnext_tiny(weights="DEFAULT")

        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)

        return model
    elif model == "mobilenet_v3_small":
        model = mobilenet_v3_small(weights="DEFAULT")
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

        return model
    else:
        return
    
def main(paths: TrainingPaths):
    
    config : Config = load_config(paths.config)
    s : TrainingSettings = config.settings
    albumentationsBuider = AlbumentationsBuilder(config.feature_engineering)
    labelLoader = LabelLoader(paths.labels)
    label_df = labelLoader.load(s.min_data_samples_per_label)
    meta = Metadata(
        dataset_size=len(label_df),
        classes=label_df.columns.to_list())
    
    early_stopping = t.EarlyStopping(
        patience=s.patience,
        min_delta=1e-4,
        save_path=s.model_path
    )

    train_df, val_df = train_test_split(
        label_df,
        test_size=s.test_size,
        random_state=42
    )
    train_dataset = t.MultiLabelDataset(
        train_df,
        paths.images,
        albumentationsBuider
    )

    val_dataset = t.MultiLabelDataset(
        val_df,
        paths.images,
        AlbumentationsBuilder({})
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    pos_weight = t.compute_pos_weights(train_df).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    sampler = t.create_sampler(train_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    model = build_model(label_df.shape[1], s.model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=s.learning_rate)

    epochs = s.epochs

    for epoch in range(epochs):

        train_loss = t.train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_loss, preds, labels = t.validate(
            model,
            val_loader,
            criterion,
            device
        )
        thresholds = t.optimize_thresholds(preds, labels)[0]
        f1 = t.compute_f1_per_class(preds, labels, thresholds=thresholds)

        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"F1 Score: {f1}")

        # early stopping check
        early_stopping(f1, train_loss, val_loss, epoch, model)

        if early_stopping.early_stop:
            break
        
    results = early_stopping.get_infos()
    save_results(paths.config, results, meta)


if __name__ == "__main__":
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent

    paths = TrainingPaths(
        config=str(ROOT / "models" / "test_7.json"),
        labels=str(ROOT / "images" / "data" / "labels.json"),
        images=str(ROOT / "images" / "data"),
    )
    main(paths)