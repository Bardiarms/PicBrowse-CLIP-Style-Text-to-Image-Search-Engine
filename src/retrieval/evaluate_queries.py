from pathlib import Path
import json
from typing import Any
import torch
from src.models.tokenizer import WordTokenizer
from src.retrieval import search


def load_eval_queries(queries_path: str) -> list[str]:
    queries_path = Path(queries_path)
    with open(queries_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_fixed_query_evaluation(
    checkpoint_path: str,
    vocab_path: str,
    image_embeddings_path: str,
    queries_path: str,
    save_path: str,
    top_k: int = 5,
    max_length: int = 32,
)-> list[dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load vocab
    vocab = torch.load(vocab_path, map_location="cpu")
    tokenizer = WordTokenizer(min_freq=1)
    tokenizer.vocab = vocab

    # load image embeddings
    image_names, image_embedding_matrix = search.load_image_embedding_cache(
        image_embeddings_path
    )

    # load model
    clip_model = search.load_trained_model_for_inference(
        checkpoint_path=checkpoint_path,
        vocab_size=len(tokenizer.vocab),
        device=device,
    )

    # load queries
    queries = load_eval_queries(queries_path)

    all_results = []

    for query in queries:
        query_embedding = search.encode_query(
            query=query,
            tokenizer=tokenizer,
            clip_model=clip_model,
            device=device,
            max_length=max_length,
        )

        results = search.search_top_k(
            query_embedding=query_embedding,
            image_names=image_names,
            image_embedding_matrix=image_embedding_matrix,
            top_k=top_k,
        )

        all_results.append({
            "query": query,
            "results": results,
        })

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    return all_results