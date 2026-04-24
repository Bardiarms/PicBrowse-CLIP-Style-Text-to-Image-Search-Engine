import torch
from src.models.tokenizer import WordTokenizer
from src.models.image_encoder import FrozenImageEncoder
from src.models.text_encoder import TextEncoder
from src.models.clip_model import PicBrowseModel
from src.retrieval.search import *
from pathlib import Path


class PicBrowseRetrievalService:
    
    def __init__(self, checkpoint_path: str,
                 vocab_path: str,
                 cached_image_embeddings_path: str,
                 image_folder_path: str
        ):
        self.vocab_path = vocab_path
        self.checkpoint_path = checkpoint_path
        self.cached_image_embeddings_path = cached_image_embeddings_path
        self.image_folder_path = Path(image_folder_path)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.vocab = torch.load(self.vocab_path, map_location="cpu")
        self.tokenizer = WordTokenizer(min_freq=1)
        self.tokenizer.vocab = self.vocab
        
        self.image_names, self.cached_image_embeddings = load_image_embedding_cache(
            image_embeddings_path=self.cached_image_embeddings_path
        )
        
        self.clip_model = load_trained_model_for_inference(
            checkpoint_path=self.checkpoint_path,
            vocab_size=len(self.vocab),
            device=self.device,
        )
        
        
        
    
    def search_query(self, query: str, top_k=5):
        query_embedding = encode_query(query=query,
                                       tokenizer= self.tokenizer,
                                       clip_model=self.clip_model,
                                       device=self.device,
                                       max_length=self.checkpoint["model_config"]["max_length"]
                        )
        results = search_top_k(query_embedding=query_embedding,
                                image_names=self.image_names,
                                image_embedding_matrix=self.cached_image_embeddings,
                                top_k=top_k
                )
        
        return results