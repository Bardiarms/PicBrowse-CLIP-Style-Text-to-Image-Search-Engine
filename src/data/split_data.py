from pathlib import Path
import random
import pandas as pd


def build_train_eval_split(
    captions_path: str,
    output_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> None:
    captions_path = Path(captions_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(captions_path)

    required_columns = {"image", "caption"}
    if not required_columns.issubset(table.columns):
        raise ValueError(
            f"Captions file must contain columns {required_columns}, got {set(table.columns)}"
        )

    unique_image_names = table["image"].astype(str).drop_duplicates().tolist()

    rng = random.Random(seed)
    rng.shuffle(unique_image_names)

    split_idx = int(len(unique_image_names) * train_ratio)

    train_image_names = set(unique_image_names[:split_idx])
    eval_image_names = set(unique_image_names[split_idx:])

    train_table = table[table["image"].astype(str).isin(train_image_names)].copy()
    eval_table = table[table["image"].astype(str).isin(eval_image_names)].copy()

    train_table.to_csv(output_dir / "train_captions.csv", index=False)
    eval_table.to_csv(output_dir / "eval_captions.csv", index=False)

    with open(output_dir / "train_images.txt", "w", encoding="utf-8") as f:
        for name in sorted(train_image_names):
            f.write(f"{name}\n")

    with open(output_dir / "eval_images.txt", "w", encoding="utf-8") as f:
        for name in sorted(eval_image_names):
            f.write(f"{name}\n")

    print(f"Total caption rows: {len(table)}")
    print(f"Total unique images: {len(unique_image_names)}")
    print(f"Train images: {len(train_image_names)}")
    print(f"Eval images: {len(eval_image_names)}")
    print(f"Train caption rows: {len(train_table)}")
    print(f"Eval caption rows: {len(eval_table)}")
    print(f"Saved split files to: {output_dir}")