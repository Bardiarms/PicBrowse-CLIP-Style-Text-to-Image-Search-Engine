from pathlib import Path
import csv


def save_training_history(loss_history: list[float], save_path: str) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])

        for epoch_idx, loss in enumerate(loss_history, start=1):
            writer.writerow([epoch_idx, loss])