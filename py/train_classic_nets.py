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
    "lenet": ModelConfig(image_size=32, grayscale=True, default_batch_size=128),
    "alexnet": ModelConfig(image_size=227, grayscale=False, default_batch_size=64),
    "vggnet": ModelConfig(image_size=224, grayscale=False, default_batch_size=16),
    "resnet": ModelConfig(image_size=224, grayscale=False, default_batch_size=64),
}


def build_transform(config):
    transform_steps = [
        transforms.Resize(256) if config.image_size >= 224 else transforms.Resize((config.image_size, config.image_size)),
    ]

    if config.image_size >= 224:
        transform_steps.append(transforms.CenterCrop(config.image_size))

    if config.grayscale:
        transform_steps.append(transforms.Grayscale(num_output_channels=1))
        transform_steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
    else:
        transform_steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    return transforms.Compose(transform_steps)


def build_model(model_name, num_classes):
    if model_name == "lenet":
        model = LeNet()
        model.dense = nn.Sequential(
            nn.Linear(120, 84),
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


def make_dataloaders(args, transform):
    train_path = args.data_root / "train"
    val_path = args.data_root / "val"

    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=transform)

    if len(train_dataset.classes) != len(val_dataset.classes):
        raise ValueError(
            "Train and val must have the same number of classes: "
            f"{len(train_dataset.classes)} != {len(val_dataset.classes)}"
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

    for step, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = loss_func(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if step % 100 == 99:
            print("[epoch:%d, step:%5d] loss: %.3f" % (epoch + 1, step + 1, running_loss / 100))
            running_loss = 0.0


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

    transform = build_transform(config)
    train_dataset, train_loader, val_loader = make_dataloaders(args, transform)

    model = build_model(args.model, num_classes=len(train_dataset.classes)).to(args.device_obj)
    print_gpu_status(args.device_obj)
    print("Model:", args.model)
    print("Batch size:", args.batch_size)
    print("Epochs:", args.epochs)
    print("Learning rate:", args.lr)
    print("Output dir:", output_dir)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        train_one_epoch(model, train_loader, optimizer, loss_func, args.device_obj, epoch)
        accuracy = evaluate(model, val_loader, args.device_obj)
        print("Accuracy of the network on the val images: %.2f %%" % accuracy)

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            torch.save(model.state_dict(), output_dir / ("weights_%d.pth" % (epoch + 1)))

    torch.save(model.state_dict(), output_dir / args.weights_name)
    print("Finished Training")


def parse_args(default_model=None):
    parser = argparse.ArgumentParser(description="Train LeNet, AlexNet, VGGNet, or ResNet on dog/cat images.")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CONFIGS),
        default=default_model,
        required=default_model is None,
        help="Model to train. Wrapper scripts set this automatically.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--weights-name", default="final_weights.pth")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--device",
        default="auto",
        help='Use "auto", "cpu", "cuda", or a specific CUDA device like "cuda:0".',
    )
    return parser.parse_args()


def main(default_model=None):
    run_training(parse_args(default_model=default_model))


if __name__ == "__main__":
    main()
