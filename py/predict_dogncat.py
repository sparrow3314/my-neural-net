import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from GoogLeNet import GoogLeNet_V1


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "dogncat"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "dogncat"


image_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


class TestImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.images = sorted(
            self.image_dir.glob("*.jpg"),
            key=lambda path: int(path.stem),
        )

        if not self.images:
            raise FileNotFoundError(f"No .jpg files found in {self.image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image_id = int(image_path.stem)

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)

        return image, image_id


def resolve_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def load_weights(model, weights_path, device):
    try:
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(weights_path, map_location=device)

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}

    model.load_state_dict(state_dict)


def predict(args):
    device = resolve_device(args.device)

    dataset = TestImageDataset(args.test_dir, transform=image_transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = GoogLeNet_V1().to(device)
    load_weights(model, args.weights, device)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    with torch.no_grad():
        for images, image_ids in loader:
            images = images.to(device)
            outputs = model(images)

            if args.output_mode == "probability":
                labels = torch.softmax(outputs, dim=1)[:, 1]
                if args.clip_eps > 0:
                    labels = labels.clamp(args.clip_eps, 1.0 - args.clip_eps)
                labels = labels.cpu().tolist()
            else:
                labels = torch.argmax(outputs, dim=1).cpu().tolist()

            for image_id, label in zip(image_ids.tolist(), labels):
                rows.append((image_id, label))

    rows.sort(key=lambda row: row[0])

    with args.output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "label"])
        if args.output_mode == "probability":
            writer.writerows((image_id, f"{label:.6f}") for image_id, label in rows)
        else:
            writer.writerows(rows)

    print(f"Saved {len(rows)} predictions to {args.output}")
    print(f"Device: {device}")


def parse_args():
    parser = argparse.ArgumentParser(description="Predict dog/cat test images.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "final_weights.pth",
        help="Path to trained model weights.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=DEFAULT_DATA_ROOT / "test",
        help="Directory containing test jpg images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "sampleSubmission.csv",
        help="Output CSV path.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--output-mode",
        choices=["probability", "class"],
        default="probability",
        help="Write dog probability for Kaggle submission, or hard 0/1 classes.",
    )
    parser.add_argument(
        "--clip-eps",
        type=float,
        default=1e-6,
        help="Clip probabilities to [eps, 1 - eps]. Use 0 to disable.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Use "auto", "cpu", "cuda", or a specific CUDA device like "cuda:0".',
    )
    return parser.parse_args()


if __name__ == "__main__":
    predict(parse_args())
