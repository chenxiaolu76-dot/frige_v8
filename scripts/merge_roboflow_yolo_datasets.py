from __future__ import annotations

import shutil
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "external_datasets"
OUTPUT_ROOT = PROJECT_ROOT / "merged_dataset_16class"
MAPPING_PATH = PROJECT_ROOT / "config" / "dataset_merge_mapping.yaml"

DATASET_DIRS = {
    "refrigerator_items": DATASET_ROOT / "refrigerator_items",
    "fridge_objects": DATASET_ROOT / "fridge_objects",
    "grocery_store_products": DATASET_ROOT / "grocery_store_products",
}

SPLIT_ALIASES = {
    "train": ["train", "images/train"],
    "val": ["val", "valid", "images/val", "images/valid"],
    "test": ["test", "images/test"],
}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def ensure_output_dirs() -> None:
    for split in ("train", "val", "test"):
        (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)


def resolve_split_dir(dataset_dir: Path, split: str, kind: str) -> Path | None:
    candidates = SPLIT_ALIASES[split]
    for candidate in candidates:
        candidate_path = dataset_dir / candidate
        if kind == "images" and candidate_path.is_dir():
            return candidate_path
        if kind == "labels":
            label_candidate = dataset_dir / candidate.replace("images/", "labels/")
            if label_candidate.is_dir():
                return label_candidate

    for candidate in candidates:
        if kind == "images":
            candidate_path = dataset_dir / candidate
        else:
            candidate_path = dataset_dir / candidate.replace("images/", "labels/")
        if candidate_path.exists():
            return candidate_path
    return None


def normalize_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def parse_dataset_names(data_yaml: dict) -> dict[int, str]:
    names = data_yaml.get("names", {})
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    return {int(idx): str(name) for idx, name in names.items()}


def write_output_yaml(target_classes: list[str]) -> None:
    data = {
        "path": str(OUTPUT_ROOT),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {idx: name for idx, name in enumerate(target_classes)},
    }
    with (OUTPUT_ROOT / "data.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)


def merge_dataset(dataset_key: str, mapping_config: dict, target_class_to_id: dict[str, int]) -> int:
    dataset_dir = DATASET_DIRS[dataset_key]
    if not dataset_dir.exists():
        print(f"[SKIP] Dataset folder not found: {dataset_dir}")
        return 0

    data_yaml_path = dataset_dir / "data.yaml"
    if not data_yaml_path.exists():
        print(f"[SKIP] data.yaml not found in: {dataset_dir}")
        return 0

    data_yaml = load_yaml(data_yaml_path)
    source_names = parse_dataset_names(data_yaml)
    source_to_target = {
        normalize_name(key): normalize_name(value)
        for key, value in mapping_config["datasets"][dataset_key]["source_to_target"].items()
    }

    merged_count = 0
    for split in ("train", "val", "test"):
        image_dir = resolve_split_dir(dataset_dir, split, "images")
        label_dir = resolve_split_dir(dataset_dir, split, "labels")
        if image_dir is None or label_dir is None:
            continue

        for label_path in label_dir.glob("*.txt"):
            new_lines: list[str] = []
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                source_cls = int(float(parts[0]))
                if source_cls not in source_names:
                    continue

                source_name = normalize_name(source_names[source_cls])
                if source_name not in source_to_target:
                    continue

                target_name = source_to_target[source_name]
                if target_name not in target_class_to_id:
                    continue

                target_cls = target_class_to_id[target_name]
                new_lines.append(" ".join([str(target_cls), *parts[1:]]))

            if not new_lines:
                continue

            image_candidates = [
                image_dir / f"{label_path.stem}.jpg",
                image_dir / f"{label_path.stem}.jpeg",
                image_dir / f"{label_path.stem}.png",
            ]
            image_path = next((candidate for candidate in image_candidates if candidate.exists()), None)
            if image_path is None:
                continue

            output_stem = f"{dataset_key}_{split}_{label_path.stem}"
            output_image_path = OUTPUT_ROOT / "images" / split / f"{output_stem}{image_path.suffix.lower()}"
            output_label_path = OUTPUT_ROOT / "labels" / split / f"{output_stem}.txt"

            shutil.copy2(image_path, output_image_path)
            output_label_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            merged_count += 1

    print(f"[DONE] {dataset_key}: merged {merged_count} labeled images")
    return merged_count


def main() -> None:
    if not MAPPING_PATH.exists():
        raise FileNotFoundError(f"Mapping file not found: {MAPPING_PATH}")

    mapping_config = load_yaml(MAPPING_PATH)
    target_classes = [normalize_name(name) for name in mapping_config["target_classes"]]
    target_class_to_id = {name: idx for idx, name in enumerate(target_classes)}

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    ensure_output_dirs()

    total = 0
    for dataset_key in DATASET_DIRS:
        total += merge_dataset(dataset_key, mapping_config, target_class_to_id)

    write_output_yaml(target_classes)
    print(f"\nMerged dataset written to: {OUTPUT_ROOT}")
    print(f"Total labeled images kept: {total}")


if __name__ == "__main__":
    main()

