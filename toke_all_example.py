# +++++
# Section 1: TinyStories Dataset Loader
# +++++
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

class TinyStoriesDataset(Dataset):
    """PyTorch Dataset for the TinyStories corpus."""
    def __init__(self, split: str = 'train'):
        self.ds = load_dataset("roneneldan/TinyStories", split=split)
        self.texts = self.ds['text']

    def __len__(self) -> int:
        """Return number of examples in the split."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        """Return the story text for the given index."""
        return self.texts[idx]

# +++++
# Section 2: Text to Vectors
# +++++
import torch
import torch.nn as nn
from transformers import AutoTokenizer

def text_to_embeddings(text: str,
                       model_name: str = "bert-base-uncased",
                       embed_dim: int = 128) -> torch.Tensor:
    """
    Tokenize `text` with a pretrained Hugging Face tokenizer,
    then map token IDs to embeddings via nn.Embedding.
    Returns: embeddings tensor of shape (1, seq_len, embed_dim)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer(text, return_tensors='pt', padding=True)
    input_ids = tokens['input_ids']       # shape: [1, seq_len]
    vocab_size = tokenizer.vocab_size

    embed_layer = nn.Embedding(vocab_size, embed_dim)
    embeddings = embed_layer(input_ids)   # shape: [1, seq_len, embed_dim]
    return embeddings

# +++++
# Section 3: Main Token Types
# +++++

def example_token_types():
    """
    Demonstrate word-level and char-level token IDs.
    """
    # Word tokens
    words = "I love NLP".split()
    word_vocab = {w: i for i, w in enumerate(set(words))}
    word_ids = torch.tensor([word_vocab[w] for w in words])
    print("Word IDs:", word_ids)

    # Char tokens
    chars = list("hello")
    char_vocab = {c: i for i, c in enumerate(set(chars))}
    char_ids = torch.tensor([char_vocab[c] for c in chars])
    print("Char IDs:", char_ids)

# +++++
# Section 4: Byte Pair Encoding (BPE)
# +++++
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

def train_and_embed_bpe(corpus: list[str], text: str, embed_dim: int = 64) -> torch.Tensor:
    """
    Train a BPE tokenizer on `corpus`, encode `text`, and map to embeddings.
    Returns: embeddings tensor of shape (1, num_subwords, embed_dim)
    """
    bpe_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(vocab_size=500, special_tokens=["[UNK]"])
    bpe_tokenizer.train_from_iterator(corpus, trainer)

    encoding = bpe_tokenizer.encode(text)
    ids = torch.tensor(encoding.ids).unsqueeze(0)
    vocab_size = bpe_tokenizer.get_vocab_size()

    embed = nn.Embedding(vocab_size, embed_dim)
    embeddings = embed(ids)
    print("Subwords:", encoding.tokens)
    print("Embeddings shape:", embeddings.shape)
    return embeddings

# +++++
# Section 5: English NLP Libraries
# +++++
def nltk_example(text: str):
    """Tokenize text using NLTK."""
    import nltk
    nltk.download('punkt', quiet=True)
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(text)
    print("NLTK tokens:", tokens)

def spacy_example(text: str):
    """Tokenize and POS-tag text using spaCy."""
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    print("spaCy tokens & POS:", [(tok.text, tok.pos_) for tok in doc])

def gensim_to_pytorch(sentences: list[list[str]]):
    """
    Train a Word2Vec on `sentences` and convert embeddings to PyTorch nn.Embedding.
    """
    from gensim.models import Word2Vec
    model = Word2Vec(sentences, vector_size=100, min_count=1)
    weights = torch.FloatTensor(model.wv.vectors)
    embedding = nn.Embedding.from_pretrained(weights)
    idx = model.wv.key_to_index[sentences[0][0]]
    print("Gensim→PyTorch vector shape:", embedding(torch.tensor(idx)).shape)

# +++++
# Section 6: tiktoken (OpenAI)
# +++++
def tiktoken_example(text: str) -> torch.Tensor:
    """
    Encode `text` using OpenAI's tiktoken for GPT-4,
    then embed IDs via nn.Embedding.
    """
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-4")
    ids = enc.encode(text)
    ids_pt = torch.tensor(ids).unsqueeze(0)
    embed = nn.Embedding(enc.n_vocab, 128)
    embeddings = embed(ids_pt)
    print("tiktoken IDs:", ids[:10], "...")
    print("Embeddings shape:", embeddings.shape)
    return embeddings

# +++++
# Section 7: Persian Tokenizers
# +++++
def persian_hazm_example(text: str) -> torch.Tensor:
    """
    Tokenize Persian text using Hazm and embed tokens.
    """
    from hazm import word_tokenize
    tokens = word_tokenize(text)
    vocab = {tok: i for i, tok in enumerate(set(tokens))}
    ids = torch.tensor([vocab[t] for t in tokens]).unsqueeze(0)
    embed = nn.Embedding(len(vocab), 64)
    embeddings = embed(ids)
    print("Hazm tokens:", tokens)
    print("Embeddings shape:", embeddings.shape)
    return embeddings

# +++++
# Section 8: Handling Unknown Tokens
# +++++
from transformers import AutoTokenizer

def unknown_token_handling_example(word: str,
                                   model_name: str = "bert-base-uncased") -> tuple[list[str], list[int]]:
    """
    Show how Hugging Face tokenizer handles completely unseen words.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.tokenize(word)
    ids    = tokenizer.convert_tokens_to_ids(tokens)
    print(f"Input word: {word!r}")
    print(" Tokens:", tokens)
    print("    IDs:", ids)
    return tokens, ids
