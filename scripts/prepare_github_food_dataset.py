from __future__ import annotations

import shutil
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "fruits-vegetables-github" / "LVIS_Fruits_And_Vegetables"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "food_subset_yolo"

# Map source class id -> target class name
TARGET_CLASS_MAP = {
    1: "apple",
    6: "banana",
    14: "carrot",
    24: "cucumber",
    43: "onion",
    44: "orange",
    52: "potato",
    8: "green_pepper",
    35: "tomato",
    59: "tomato",
}

TARGET_CLASS_ORDER = [
    "apple",
    "banana",
    "orange",
    "tomato",
    "cucumber",
    "potato",
    "onion",
    "carrot",
    "green_pepper",
]

TARGET_CLASS_ID = {name: idx for idx, name in enumerate(TARGET_CLASS_ORDER)}


def ensure_dirs() -> None:
    for split in ("train", "val", "test"):
        (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)


def convert_split(split: str, source_image_split: str) -> int:
    image_dir = RAW_ROOT / "images" / source_image_split
    label_dir = RAW_ROOT / "labels" / split
    output_image_dir = OUTPUT_ROOT / "images" / split
    output_label_dir = OUTPUT_ROOT / "labels" / split

    kept_images = 0
    for label_path in label_dir.glob("*.txt"):
        new_lines: list[str] = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split()
            source_cls = int(float(parts[0]))
            if source_cls not in TARGET_CLASS_MAP:
                continue
            target_name = TARGET_CLASS_MAP[source_cls]
            target_cls = TARGET_CLASS_ID[target_name]
            new_lines.append(" ".join([str(target_cls), *parts[1:]]))

        if not new_lines:
            continue

        image_path = image_dir / f"{label_path.stem}.jpg"
        if not image_path.exists():
            continue

        shutil.copy2(image_path, output_image_dir / image_path.name)
        (output_label_dir / label_path.name).write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        kept_images += 1

    return kept_images


def write_data_yaml() -> None:
    data = {
        "path": str(OUTPUT_ROOT),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {idx: name for idx, name in enumerate(TARGET_CLASS_ORDER)},
    }
    with (OUTPUT_ROOT / "data.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)


def main() -> None:
    ensure_dirs()
    train_count = convert_split("train", "train")
    val_count = convert_split("val", "val")
    test_count = convert_split("test", "test")
    write_data_yaml()
    print(
        f"Prepared filtered dataset at {OUTPUT_ROOT}\n"
        f"train images: {train_count}\n"
        f"val images: {val_count}\n"
        f"test images: {test_count}"
    )


if __name__ == "__main__":
    main()
