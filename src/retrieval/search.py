import torch
from typing import Tuple
from src.models.tokenizer import WordTokenizer
from src.models.clip_model import PicBrowseModel
from src.models.image_encoder import FrozenImageEncoder
from src.models.text_encoder import TextEncoder


def load_image_embedding_cache(image_embeddings_path: str
                            )-> Tuple:
    cache = torch.load(image_embeddings_path)
    
    image_names = sorted(cache.keys())
    embedding_matrix = torch.stack([cache[name] for name in image_names], dim=0)

    return image_names, embedding_matrix


def load_trained_model_for_inference(
    checkpoint_path: str,
    vocab_size: int,
    device: torch.device,
    max_length: int = 32,
    embed_dim: int = 256,
    num_heads: int = 4,
    num_layers: int = 2,
    ff_dim: int = 512,
    output_dim: int = 512,
    )-> PicBrowseModel:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    image_encoder = FrozenImageEncoder()
    text_encoder = TextEncoder(
        vocab_size=vocab_size,
        max_length=max_length,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
        output_dim=output_dim,
    )

    clip_model = PicBrowseModel(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        init_logit_scale=0.0,
    )

    clip_model.load_state_dict(checkpoint["model_state_dict"])
    clip_model.to(device)
    clip_model.eval()

    return clip_model


def encode_query(
    query: str,
    tokenizer: WordTokenizer,
    clip_model: PicBrowseModel,
    device: torch.device,
    max_length: int = 32,
    )-> torch.Tensor:
    encoded = tokenizer.encode_plus(query, max_length=max_length)

    input_ids = torch.tensor(
        [encoded["input_ids"]],
        dtype=torch.long,
        device=device
    )

    attention_mask = torch.tensor(
        [encoded["attention_mask"]],
        dtype=torch.long,
        device=device
    )

    clip_model.eval()
    with torch.no_grad():
        query_embedding = clip_model.encode_text(input_ids, attention_mask)

    return query_embedding.cpu()


def search_top_k(
    query_embedding: torch.Tensor,
    image_names: list[str],
    image_embedding_matrix: torch.Tensor,
    top_k: int = 5,
):
    image_embedding_matrix = image_embedding_matrix.to(query_embedding.device)

    scores = query_embedding @ image_embedding_matrix.T
    scores = scores.squeeze(0)

    top_scores, top_indices = torch.topk(scores, k=top_k)

    results = []
    for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
        results.append({
            "file_name": image_names[idx],
            "score": score,
        })

    return results