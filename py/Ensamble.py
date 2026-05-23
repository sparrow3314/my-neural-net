import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from GoogLeNet import GoogLeNet_V1
from predict_dogncat import load_weights
from train_classic_nets import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    MODEL_CONFIGS,
    build_model,
    build_transform,
    canonical_class_to_idx,
    resolve_device,
)


AVAILABLE_MODELS = ("alexnet", "googlenet", "resnet", "vggnet")


def get_model(model_name):
    if model_name == "googlenet":
        return GoogLeNet_V1()
    return build_model(model_name, num_classes=2)


def get_transform(model_name):
    if model_name == "googlenet":
        return build_transform(MODEL_CONFIGS["resnet"])
    return build_transform(MODEL_CONFIGS[model_name])


def get_weights_paths(weights_root, weights_name, model_names):
    return {
        model_name: weights_root / model_name / weights_name
        for model_name in model_names
    }


def validate_weights_exist(weights_paths):
    missing_paths = [
        str(weights_path)
        for weights_path in weights_paths.values()
        if not weights_path.is_file()
    ]
    if missing_paths:
        raise FileNotFoundError("Missing model weights:\n" + "\n".join(missing_paths))


def validate_class_mapping(data_root):
    train_dataset = datasets.ImageFolder(root=data_root / "train")
    val_dataset = datasets.ImageFolder(root=data_root / "val")

    train_class_to_idx = canonical_class_to_idx(train_dataset)
    val_class_to_idx = canonical_class_to_idx(val_dataset)
    if train_class_to_idx != val_class_to_idx:
        raise ValueError(
            "Train and val class mappings must match: "
            f"{train_dataset.class_to_idx} != {val_dataset.class_to_idx}"
        )

    cat_idx = val_class_to_idx.get("cat")
    dog_idx = val_class_to_idx.get("dog")
    if cat_idx is None or dog_idx is None:
        raise ValueError(f"Could not find cat/dog classes in val mapping: {val_dataset.class_to_idx}")

    return cat_idx, dog_idx


def build_val_datasets(data_root, model_names):
    val_path = data_root / "val"
    val_datasets = {
        model_name: datasets.ImageFolder(root=val_path, transform=get_transform(model_name))
        for model_name in model_names
    }
    validate_sample_order(val_datasets, model_names)
    return val_datasets


def validate_sample_order(val_datasets, model_names):
    first_name = model_names[0]
    expected_samples = [
        (Path(path).resolve(), label)
        for path, label in val_datasets[first_name].samples
    ]

    for model_name in model_names[1:]:
        samples = [
            (Path(path).resolve(), label)
            for path, label in val_datasets[model_name].samples
        ]
        if samples != expected_samples:
            raise ValueError(
                f"Val sample order differs between {first_name} and {model_name}"
            )


def predict_model(model_name, dataset, weights_path, device, batch_size, num_workers, cat_idx, dog_idx):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = get_model(model_name).to(device)
    load_weights(model, weights_path, device)
    model.eval()

    probabilities = []
    labels = []

    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            dog_probabilities = torch.softmax(outputs, dim=1)[:, dog_idx]
            probabilities.append(dog_probabilities.cpu())
            labels.append(batch_labels.cpu())

    probabilities = torch.cat(probabilities)
    labels = torch.cat(labels)
    predictions = torch.where(
        probabilities >= 0.5,
        torch.full_like(labels, dog_idx),
        torch.full_like(labels, cat_idx),
    )
    correct = (predictions == labels).sum().item()
    accuracy = 100.0 * correct / labels.numel() if labels.numel() else 0.0
    return probabilities, labels, accuracy


def resolve_ensemble_weights(model_names, weights):
    if weights is None:
        return torch.full((len(model_names),), 1.0 / len(model_names), dtype=torch.float32)

    if len(weights) != len(model_names):
        raise ValueError(
            "Number of ensemble weights must match selected models: "
            f"{len(weights)} weights for {len(model_names)} models"
        )

    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    if torch.any(weight_tensor < 0):
        raise ValueError(f"Ensemble weights must be non-negative: {weights}")

    total_weight = weight_tensor.sum().item()
    if total_weight <= 0:
        raise ValueError(f"At least one ensemble weight must be positive: {weights}")

    return weight_tensor / total_weight


def run_ensemble(args):
    device = resolve_device(args.device)
    model_names = tuple(args.models)
    ensemble_weights = resolve_ensemble_weights(model_names, args.ensemble_weights)
    weights_paths = get_weights_paths(args.weights_root, args.weights_name, model_names)
    validate_weights_exist(weights_paths)
    cat_idx, dog_idx = validate_class_mapping(args.data_root)
    val_datasets = build_val_datasets(args.data_root, model_names)

    model_probabilities = []
    reference_labels = None
    model_accuracies = {}

    for model_name in model_names:
        probabilities, labels, accuracy = predict_model(
            model_name,
            val_datasets[model_name],
            weights_paths[model_name],
            device,
            args.batch_size,
            args.num_workers,
            cat_idx,
            dog_idx,
        )
        if reference_labels is None:
            reference_labels = labels
        elif not torch.equal(reference_labels, labels):
            raise ValueError(f"Val labels differ for model: {model_name}")

        model_probabilities.append(probabilities)
        model_accuracies[model_name] = accuracy

    probability_stack = torch.stack(model_probabilities, dim=0)
    average_probabilities = (probability_stack * ensemble_weights[:, None]).sum(dim=0)
    ensemble_predictions = torch.where(
        average_probabilities >= 0.5,
        torch.full_like(reference_labels, dog_idx),
        torch.full_like(reference_labels, cat_idx),
    )
    ensemble_correct = (ensemble_predictions == reference_labels).sum().item()
    ensemble_accuracy = 100.0 * ensemble_correct / reference_labels.numel()

    print("Model accuracies:")
    for model_name, weight in zip(model_names, ensemble_weights.tolist()):
        print(f"{model_name}: {model_accuracies[model_name]:.2f} % | weight: {weight:.4f}")
    print()
    print(f"Ensemble accuracy: {ensemble_accuracy:.2f} %")
    print(f"Device: {device}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate probability-averaged ensemble on dog/cat val images."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--weights-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--weights-name", default="final_weights.pth")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=AVAILABLE_MODELS,
        default=AVAILABLE_MODELS,
        help="Models to include in the ensemble.",
    )
    parser.add_argument(
        "--weights",
        "--ensemble-weights",
        dest="ensemble_weights",
        nargs="+",
        type=float,
        default=None,
        help="Optional ensemble weights aligned with --models order. Values are normalized automatically.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help='Use "auto", "cpu", "cuda", or a specific CUDA device like "cuda:0".',
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_ensemble(parse_args())
