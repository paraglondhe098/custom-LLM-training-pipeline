import torch
import os
from typing import Dict, Optional
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
import re
from torch.utils.data import Dataset


class TextProcessor:
    def __init__(self, path: str):
        self.path = path

    def combine(self, ret: bool = False) -> Optional[str]:
        """
        Combine text from all .txt files in the directory.
        Args:
            ret (bool): Whether to return the combined text
        Returns:
            Combined text if ret is True, else None
        """
        corpus = []
        for file_name in os.listdir(self.path):
            if file_name.endswith(".txt"):
                with open(os.path.join(self.path, file_name), "r", encoding="utf-8") as file:
                    content = file.read()
                    content = self.preprocess(content)
                    corpus.append(content)
        return "\n".join(corpus) if ret else None

    @staticmethod
    def preprocess(text: str) -> str:
        """
        Preprocess the text by cleaning and normalizing it.
        Args:
            text (str): The raw text to preprocess
        Returns:
            str: The cleaned and normalized text
        """
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text.strip())
        # Convert text to lowercase
        text = text.lower()
        # Remove special characters
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text

    def create_corpus(self, corpus_file: str) -> None:
        """
        Preprocess the text data and save the combined corpus.
        Args:
            corpus_file (str): Path to save the corpus
        """
        combined_text = self.combine(ret=True)
        with open(corpus_file, "w", encoding="utf-8") as file:
            file.write(combined_text)
        print(f"Corpus saved to {corpus_file}")


class TokenizeText:
    def __init__(self, corpus_file: str, tokenizer_save_path: str, mode: str = "train"):
        """
        Initialize tokenizer, train on corpus, and save.
        Args:
            corpus_file (str): Path to the corpus file
            tokenizer_save_path (str): Path to save the tokenizer
        """
        self.corpus_file = corpus_file
        self.save_path = tokenizer_save_path
        if mode == "train":
            self.tokenizer = self._train_and_save()
        else:
            self.tokenizer = self.load_tokenizer()

    def _train_and_save(self) -> PreTrainedTokenizerFast:
        """
        Train BPE tokenizer and save in Hugging Face format.
        Returns:
            Hugging Face tokenizer
        """
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = Whitespace()

        trainer = BpeTrainer(
            special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
            min_frequency=2  # Reduce vocab size
        )
        tokenizer.train(files=[self.corpus_file], trainer=trainer)

        # Save tokenizer
        tokenizer_json_path = os.path.join(self.save_path, "custom_tokenizer.json")
        tokenizer.save(tokenizer_json_path)

        # Wrap with Hugging Face tokenizer
        hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_json_path)
        hf_tokenizer.save_pretrained(self.save_path)
        print(f"Tokenizer trained and saved to {self.save_path}")

        return hf_tokenizer

    def load_tokenizer(self):
        tokenizer_json_path = os.path.join(self.save_path, "custom_tokenizer.json")
        if not os.path.exists(tokenizer_json_path):
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_json_path}")

        hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_json_path)
        print(f"Tokenizer loaded from {tokenizer_json_path}")
        return hf_tokenizer

    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize input text."""
        return self.tokenizer(text, return_tensors="pt")

    def untokenize(self, tokens: torch.Tensor) -> str:
        """Convert tokens back into text."""
        return self.tokenizer.decode(tokens)


class TextDataset(Dataset):
    def __init__(self, tokenizer, corpus_file, block_size, random=False, split="train", split_ratio=0.9):
        self.tokenizer = tokenizer
        self.random = random
        self.corpus_file = corpus_file
        self.block_size = block_size

        with open(corpus_file, 'r', encoding='utf-8') as f:
            texts = f.read()

        encoding = self.tokenizer.tokenize(texts)
        self.df = encoding['input_ids'].squeeze()
        if split == "train":
            self.df = self.df[:int(split_ratio * len(self.df))]
        else:
            self.df = self.df[int(split_ratio * len(self.df)):]

    def __len__(self):
        return len(self.df) - self.block_size

    def __getitem__(self, idx):
        if self.random:
            idx = torch.randint(low=0, high=len(self), size=(1,))
        x = self.df[idx:idx + self.block_size]
        y = self.df[idx + 1:idx + self.block_size + 1]
        return x, y
