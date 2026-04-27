from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


def show_saved_eval_result(
    eval_results: list[dict],
    image_folder: str,
    query_index: int = 0,
    max_results: int = 5,
):
    item = eval_results[query_index]
    query = item["query"]
    results = item["results"][:max_results]

    image_folder = Path(image_folder)

    plt.figure(figsize=(4 * len(results), 5))

    for i, result in enumerate(results, start=1):
        image_path = image_folder / result["file_name"]

        if not image_path.exists():
            print(f"Missing image: {image_path}")
            continue

        image = Image.open(image_path).convert("RGB")

        plt.subplot(1, len(results), i)
        plt.imshow(image)
        plt.axis("off")
        plt.title(f'{result["file_name"]}\nscore={result["score"]:.4f}')

    plt.suptitle(f"Query: {query}", fontsize=14)
    plt.tight_layout()
    plt.show()