from pydantic import BaseModel
import json
from pathlib import Path

class TrainingPaths(BaseModel):
    config: str
    labels: str
    images: str

class TrainingSettings(BaseModel):
    epochs: int
    test_size: float
    learning_rate: float
    patience: int
    min_data_samples_per_label: int
    batch_size: int
    model_path: str
    model: str
    max_pos_weight: float = 10.0
    
class TrainingResults(BaseModel):
    val_f1: float = 0
    val_class_f1: list[float] | None = []
    train_loss: float = 0
    val_loss: float = 0
    stop_epcoh: int = 0

class Metadata(BaseModel):
    dataset_size: int = 0
    classes: list = []

class MultipleMeta(BaseModel):
    iteration: dict[str, Metadata]
    
class Config(BaseModel):
    notes: str
    settings: TrainingSettings
    results: TrainingResults
    meta: Metadata
    feature_engineering: dict

def load_config(file):
    with open(file, "r") as f:
        data = json.load(f)
    return Config.model_validate(data)

def save_results(file, results: TrainingResults, meta: Metadata |MultipleMeta):
    with open(file, "r") as f:
        data = json.load(f)
    
    data["results"] = results.model_dump()
    data["meta"] = meta.model_dump()

    Path(file).write_text(json.dumps(data, indent=4))