from pathlib import Path
import shutil
from typing import Dict, List, Optional, Tuple

from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = BASE_DIR / "best.pt"
DEFAULT_LINEAR_WEIGHTS = BASE_DIR / "linear_classifier.pt"
CLASSES = ["helmet", "no_helmet", "special_case"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

LINEAR_TRANSFORM = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def resolve_dataset_dir(base_dir: Path = BASE_DIR) -> Path:
    """Resolve local dataset directory ('dataset' or 'datasets') containing train/ and val/."""
    for name in ("dataset", "datasets"):
        candidate = base_dir / name
        if (candidate / "train").is_dir() and (candidate / "val").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find 'dataset/' or 'datasets/' with train/ and val/ in {base_dir}"
    )


def _build_effnet_backbone(pretrained: bool = False) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    eff = models.efficientnet_b0(weights=weights)
    backbone = nn.Sequential(eff.features, eff.avgpool, nn.Flatten())
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


def load_linear_classifier(
    weights_path: Optional[Path] = None,
) -> Tuple[nn.Module, nn.Linear, List[str]]:
    """Load self-contained frozen EfficientNet-B0 backbone + trained linear head from disk."""
    target = Path(weights_path) if weights_path else DEFAULT_LINEAR_WEIGHTS
    ckpt = torch.load(str(target), map_location="cpu", weights_only=True)
    classes = list(ckpt.get("classes", CLASSES))
    backbone = _build_effnet_backbone(pretrained=False)
    backbone.load_state_dict(ckpt["backbone_state"])
    backbone.eval()

    head = nn.Linear(1280, len(classes))
    head.load_state_dict(ckpt["head_state"])
    head.eval()
    return backbone, head, classes


def predict_linear_probs(
    backbone: nn.Module, head: nn.Linear, rgb_images: List[Image.Image]
) -> torch.Tensor:
    """Return softmax probabilities (N, C) for a list of RGB PIL images."""
    batch = torch.stack([LINEAR_TRANSFORM(img) for img in rgb_images])
    with torch.no_grad():
        emb = nn.functional.normalize(backbone(batch), dim=1)
        return torch.softmax(head(emb), dim=1)


def verify_linear_accuracy(weights_path: Path, val_dir: Path) -> Dict[str, float]:
    """Evaluate frozen-backbone + balanced linear classifier on val/ subfolders."""
    backbone, head, classes = load_linear_classifier(weights_path)
    per_class_acc: Dict[str, float] = {}
    total_correct = 0
    total_images = 0

    print("\n=== Frozen Feature Extractor + Linear Classifier (val/) ===")
    for ci, cls_name in enumerate(classes):
        class_dir = val_dir / cls_name
        if not class_dir.is_dir():
            continue
        images = [
            p for p in sorted(class_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        if not images:
            continue

        pil_imgs = [Image.open(p).convert("RGB") for p in images]
        probs = predict_linear_probs(backbone, head, pil_imgs)
        preds = probs.argmax(dim=1)
        correct = int((preds == ci).sum().item())
        acc = (correct / len(images)) * 100.0
        per_class_acc[cls_name] = acc
        total_correct += correct
        total_images += len(images)
        print(f"  {cls_name:14s}: {correct:2d}/{len(images):2d} ({acc:6.2f}%)")

    overall = (total_correct / total_images * 100.0) if total_images else 0.0
    print(f"  {'OVERALL':14s}: {total_correct:2d}/{total_images:2d} ({overall:6.2f}%)")
    print("===========================================================\n")
    return per_class_acc


def ensure_linear_classifier(
    weights_path: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """Train frozen EfficientNet-B0 feature extractor + balanced linear head if missing or forced."""
    target = Path(weights_path) if weights_path else DEFAULT_LINEAR_WEIGHTS
    if target.is_file() and not force:
        return target

    dataset_root = Path(data_dir) if data_dir else resolve_dataset_dir()
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"

    print(f"[train_model] Extracting frozen EfficientNet-B0 features from '{train_dir}'...")
    backbone = _build_effnet_backbone(pretrained=True)

    aug_color = T.Compose(
        [
            T.Resize((224, 224)),
            T.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    torch.manual_seed(42)
    feats, labels = [], []
    for ci, cls_name in enumerate(CLASSES):
        cls_dir = train_dir / cls_name
        if not cls_dir.is_dir():
            continue
        for p in sorted(cls_dir.iterdir()):
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
                continue
            img = Image.open(p).convert("RGB")
            flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
            batch = torch.stack(
                [
                    LINEAR_TRANSFORM(img),
                    LINEAR_TRANSFORM(flipped),
                    aug_color(img),
                    aug_color(flipped),
                ]
            )
            with torch.no_grad():
                emb = nn.functional.normalize(backbone(batch), dim=1)
            feats.append(emb)
            labels.extend([ci] * len(batch))

    F_tr = torch.cat(feats, dim=0)
    y_tr = torch.tensor(labels, dtype=torch.long)

    counts = torch.bincount(y_tr, minlength=len(CLASSES)).float()
    class_weights = counts.sum() / (len(CLASSES) * counts.clamp(min=1.0))

    head = nn.Linear(1280, len(CLASSES))
    opt = torch.optim.AdamW(head.parameters(), lr=0.05, weight_decay=0.02)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    for _ in range(320):
        opt.zero_grad()
        loss = loss_fn(head(F_tr), y_tr)
        loss.backward()
        opt.step()

    torch.save(
        {
            "backbone_state": backbone.state_dict(),
            "head_state": head.state_dict(),
            "classes": CLASSES,
        },
        str(target),
    )
    print(f"[train_model] Saved self-contained linear classifier to: {target}")
    if val_dir.is_dir():
        verify_linear_accuracy(target, val_dir)
    return target


def verify_validation_accuracy(weights_path: Path, val_dir: Path) -> Dict[str, float]:
    """Run YOLO inference over val/ subfolders and print per-class validation accuracy."""
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    per_class_acc: Dict[str, float] = {}
    total_correct = 0
    total_images = 0

    print("\n=== YOLO Validation Accuracy Verification (val/) ===")
    for class_dir in sorted(p for p in val_dir.iterdir() if p.is_dir()):
        true_cls = class_dir.name
        images = [
            p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        if not images:
            continue

        correct = 0
        for img_path in images:
            res = model(str(img_path), imgsz=224, verbose=False)[0]
            pred_idx = int(res.probs.top1)
            pred_cls = str(res.names[pred_idx])
            if pred_cls == true_cls:
                correct += 1

        acc = (correct / len(images)) * 100.0
        per_class_acc[true_cls] = acc
        total_correct += correct
        total_images += len(images)
        print(f"  {true_cls:14s}: {correct:2d}/{len(images):2d} ({acc:6.2f}%)")

    overall = (total_correct / total_images * 100.0) if total_images else 0.0
    print(f"  {'OVERALL':14s}: {total_correct:2d}/{total_images:2d} ({overall:6.2f}%)")
    print("==================================================\n")
    return per_class_acc


def ensure_trained_model(
    weights_path: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    epochs: int = 35,
    imgsz: int = 224,
    force: bool = False,
) -> Path:
    """Train YOLO11n-cls on the local helmet/no_helmet/special_case dataset if best.pt is missing or forced."""
    target_weights = Path(weights_path) if weights_path else DEFAULT_WEIGHTS
    dataset_root = Path(data_dir) if data_dir else resolve_dataset_dir()

    if target_weights.is_file() and not force:
        return target_weights

    print(f"[train_model] Training YOLO('yolo11n-cls.pt') on '{dataset_root}'...")
    from ultralytics import YOLO

    base_model = YOLO("yolo11n-cls.pt")
    results = base_model.train(
        data=str(dataset_root),
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        project=str(BASE_DIR / "runs_cls"),
        name="helmet_cls",
        exist_ok=True,
    )

    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else (BASE_DIR / "runs_cls" / "helmet_cls")
    trained_best = save_dir / "weights" / "best.pt"
    if not trained_best.is_file():
        trained_best = save_dir / "weights" / "last.pt"
    if not trained_best.is_file():
        raise FileNotFoundError(f"Expected trained weights not found under {save_dir / 'weights'}")

    shutil.copy2(trained_best, target_weights)
    print(f"[train_model] Saved trained weights to: {target_weights}")

    verify_validation_accuracy(target_weights, dataset_root / "val")
    return target_weights


if __name__ == "__main__":
    ensure_linear_classifier(force=True)
