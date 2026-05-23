import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ResNet import ResNet18
from net import Alexnet, LeNet, VGGnet


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "dogncat"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "dogncat"


@dataclass(frozen=True)
class ModelConfig:
    image_size: int
    grayscale: bool
    default_batch_size: int
    default_lr: float = 0.004


MODEL_CONFIGS = {
    "lenet": ModelConfig(image_size=64, grayscale=True, default_batch_size=128),
    "alexnet": ModelConfig(image_size=227, grayscale=False, default_batch_size=64),
    "vggnet": ModelConfig(image_size=224, grayscale=False, default_batch_size=16),
    "resnet": ModelConfig(image_size=224, grayscale=False, default_batch_size=64),
}


CLASS_ALIASES = {
    "cat": "cat",
    "cats": "cat",
    "dog": "dog",
    "dogs": "dog",
}


def normalize_class_name(class_name):
    normalized = class_name.strip().lower()
    return CLASS_ALIASES.get(normalized, normalized)


def canonical_class_to_idx(dataset):
    return {normalize_class_name(class_name): idx for class_name, idx in dataset.class_to_idx.items()}


def build_normalize_steps(config):
    if config.grayscale:
        return [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]

    return [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]


def build_transforms(config):
    if config.image_size >= 224:
        base_steps = [
            transforms.Resize(256),
            transforms.CenterCrop(config.image_size),
        ]
    else:
        base_steps = [
            transforms.Resize((config.image_size, config.image_size)),
        ]

    train_transform = transforms.Compose(base_steps + build_normalize_steps(config))
    val_transform = transforms.Compose(base_steps + build_normalize_steps(config))
    return train_transform, val_transform


def build_transform(config):
    _, val_transform = build_transforms(config)
    return val_transform


def build_model(model_name, num_classes):
    if model_name == "lenet":
        model = LeNet()
        model.dense = nn.Sequential(
            nn.Linear(model.flatten_features, 84),
            # nn.ReLU(inplace=True),
            nn.Sigmoid(),
            nn.Linear(84, num_classes),
        )
        return model

    if model_name == "alexnet":
        model = Alexnet()
        model.dense = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
        )
        return model

    if model_name == "vggnet":
        model = VGGnet()
        model.dense = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
        )
        return model

    if model_name == "resnet":
        model = ResNet18()
        model.dense = nn.Linear(512, num_classes)
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def resolve_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def print_gpu_status(device):
    use_gpu = device.type == "cuda"
    print("CUDA available:", torch.cuda.is_available())
    print("Training device:", device)
    print("GPU enabled for training:", use_gpu)

    if not torch.cuda.is_available():
        print("No CUDA GPU detected; training will run on CPU.")
        return

    print("CUDA version:", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())
    current_device = torch.cuda.current_device()

    for device_id in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(device_id)
        total_memory = props.total_memory / (1024**3)
        allocated_memory = torch.cuda.memory_allocated(device_id) / (1024**3)
        reserved_memory = torch.cuda.memory_reserved(device_id) / (1024**3)
        marker = "*" if use_gpu and device_id == current_device else " "
        print(
            "%s GPU %d: %s | total: %.2f GB | allocated: %.2f GB | reserved: %.2f GB"
            % (marker, device_id, props.name, total_memory, allocated_memory, reserved_memory)
        )


def make_dataloaders(args, train_transform, val_transform):
    train_path = args.data_root / "train"
    val_path = args.data_root / "val"

    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=val_transform)

    if len(train_dataset.classes) != len(val_dataset.classes):
        raise ValueError(
            "Train and val must have the same number of classes: "
            f"{len(train_dataset.classes)} != {len(val_dataset.classes)}"
        )
    train_class_to_idx = canonical_class_to_idx(train_dataset)
    val_class_to_idx = canonical_class_to_idx(val_dataset)
    if train_class_to_idx != val_class_to_idx:
        raise ValueError(
            "Train and val class mappings must match: "
            f"{train_dataset.class_to_idx} != {val_dataset.class_to_idx}"
        )
    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        print(
            "Class folder names differ but normalized mappings match: "
            f"{train_dataset.class_to_idx} -> {train_class_to_idx}, "
            f"{val_dataset.class_to_idx} -> {val_class_to_idx}"
        )

    pin_memory = args.device_obj.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    print("Train classes:", train_dataset.class_to_idx)
    print("Val classes:", val_dataset.class_to_idx)
    return train_dataset, train_loader, val_loader


def train_one_epoch(model, loader, optimizer, loss_func, device, epoch):
    model.train()
    running_loss = 0.0
    total_loss = 0.0
    correct = 0
    total = 0

    for step, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        batch_size = labels.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_loss += loss.item() * batch_size
        _, predicted = torch.max(outputs.data, 1)
        total += batch_size
        correct += (predicted == labels).sum().item()
        if step % 100 == 99:
            print("[epoch:%d, step:%5d] loss: %.3f" % (epoch + 1, step + 1, running_loss / 100))
            running_loss = 0.0

    avg_loss = total_loss / total if total else 0.0
    accuracy = 100.0 * correct / total if total else 0.0
    return avg_loss, accuracy


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100.0 * correct / total if total else 0.0


def build_optimizer(args, model):
    if args.optimizer == "adam":
        return optim.Adam(model.parameters(), lr=args.lr)
    if args.optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def run_training(args):
    config = MODEL_CONFIGS[args.model]
    if args.batch_size is None:
        args.batch_size = config.default_batch_size
    if args.lr is None:
        args.lr = config.default_lr

    args.device_obj = resolve_device(args.device)
    output_dir = args.output_root / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    train_transform, val_transform = build_transforms(config)
    train_dataset, train_loader, val_loader = make_dataloaders(args, train_transform, val_transform)

    model = build_model(args.model, num_classes=len(train_dataset.classes)).to(args.device_obj)
    print_gpu_status(args.device_obj)
    print("Model:", args.model)
    print("Batch size:", args.batch_size)
    print("Epochs:", args.epochs)
    print("Learning rate:", args.lr)
    print("Optimizer:", args.optimizer)
    print("LR reduce factor:", args.lr_reduce_factor)
    print("LR patience:", args.lr_patience)
    print("Minimum learning rate:", args.min_lr)
    print("Output dir:", output_dir)

    optimizer = build_optimizer(args, model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_reduce_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, loss_func, args.device_obj, epoch)
        accuracy = evaluate(model, val_loader, args.device_obj)
        scheduler.step(accuracy)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            "Epoch %d/%d | train loss: %.4f | train acc: %.2f %% | val acc: %.2f %% | lr: %.6f"
            % (epoch + 1, args.epochs, train_loss, train_accuracy, accuracy, current_lr)
        )

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            torch.save(model.state_dict(), output_dir / ("weights_%d.pth" % (epoch + 1)))

    torch.save(model.state_dict(), output_dir / args.weights_name)
    print("Finished Training")


def parse_args(
    default_model=None,
    default_epochs=50,
    default_batch_size=None,
    default_lr=None,
    default_momentum=0.9,
    default_optimizer="sgd",
    default_num_workers=4,
    default_lr_reduce_factor=0.5,
    default_lr_patience=3,
    default_min_lr=1e-5,
    default_save_every=0,
    default_weights_name="final_weights.pth",
    default_seed=None,
    default_device="auto",
    default_data_root=DEFAULT_DATA_ROOT,
    default_output_root=DEFAULT_OUTPUT_ROOT,
):
    parser = argparse.ArgumentParser(description="Train LeNet, AlexNet, VGGNet, or ResNet on dog/cat images.")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CONFIGS),
        default=default_model,
        required=default_model is None,
        help="Model to train. Wrapper scripts set this automatically.",
    )
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument("--output-root", type=Path, default=default_output_root)
    parser.add_argument("--epochs", type=int, default=default_epochs)
    parser.add_argument("--batch-size", type=int, default=default_batch_size)
    parser.add_argument("--lr", type=float, default=default_lr)
    parser.add_argument("--momentum", type=float, default=default_momentum)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default=default_optimizer)
    parser.add_argument("--num-workers", type=int, default=default_num_workers)
    parser.add_argument("--lr-reduce-factor", type=float, default=default_lr_reduce_factor)
    parser.add_argument("--lr-patience", type=int, default=default_lr_patience)
    parser.add_argument("--min-lr", type=float, default=default_min_lr)
    parser.add_argument("--save-every", type=int, default=default_save_every)
    parser.add_argument("--weights-name", default=default_weights_name)
    parser.add_argument("--seed", type=int, default=default_seed)
    parser.add_argument(
        "--device",
        default=default_device,
        help='Use "auto", "cpu", "cuda", or a specific CUDA device like "cuda:0".',
    )
    return parser.parse_args()


def main(default_model=None, **default_kwargs):
    run_training(parse_args(default_model=default_model, **default_kwargs))


if __name__ == "__main__":
    main()
