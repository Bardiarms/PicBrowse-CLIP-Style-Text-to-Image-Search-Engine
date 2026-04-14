import re
from collections import Counter


class WordTokenizer:
    
    def __init__(self, min_freq = 1):
        self.min_freq = min_freq
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"

    def clean_text(self, text: str)-> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)   # remove punctuation
        text = re.sub(r"\s+", " ", text)      # collapse repeated spaces
        return text

    def tokenize(self, text: str)-> list:
        text = self.clean_text(text)
        return text.split()


    def build_vocab(self, texts):
        counter = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)
        
        for token, freq in counter.items():
            if freq >= self.min_freq and token not in self.vocab:
                self.vocab[token] = len(self.vocab)

    def encode(self, text)-> list:
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]

    def encode_plus(self, text, max_length=32)-> dict:
        ids = self.encode(text)
        ids = ids[:max_length]      # Truncate if the length is larger than max_length

        attention_mask = [1] * len(ids)

        while len(ids) < max_length:        # Add "0" if the length is lower than max_length
            ids.append(self.vocab[self.pad_token])
            attention_mask.append(0)

        return {
            "input_ids": ids,
            "attention_mask": attention_mask
        }
        
        
