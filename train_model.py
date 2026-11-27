from pathlib import Path
import shutil
from typing import Dict, Optional

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = BASE_DIR / "best.pt"


def resolve_dataset_dir(base_dir: Path = BASE_DIR) -> Path:
    """Resolve local dataset directory ('dataset' or 'datasets') containing train/ and val/."""
    for name in ("dataset", "datasets"):
        candidate = base_dir / name
        if (candidate / "train").is_dir() and (candidate / "val").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find 'dataset/' or 'datasets/' with train/ and val/ in {base_dir}"
    )


def verify_validation_accuracy(weights_path: Path, val_dir: Path) -> Dict[str, float]:
    """Run inference over val/ subfolders and print per-class validation accuracy."""
    model = YOLO(str(weights_path))
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    per_class_acc: Dict[str, float] = {}
    total_correct = 0
    total_images = 0

    print("\n=== Validation Accuracy Verification (val/) ===")
    for class_dir in sorted(p for p in val_dir.iterdir() if p.is_dir()):
        true_cls = class_dir.name
        images = [
            p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts
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
    print("=============================================\n")
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
    ensure_trained_model(force=True)
