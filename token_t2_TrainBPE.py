from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from datasets import load_dataset

class BPETokenizer:
    def __init__(self, vocab_size: int = 1000, unk_token: str = "[UNK]"):
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        # initialize an empty BPE tokenizer
        self.tokenizer = Tokenizer(BPE(unk_token=self.unk_token))

    def train(self, texts_iterator):
        """
        texts_iterator: any iterable of strings (e.g. a list or dataset column)
        """
        trainer = BpeTrainer(vocab_size=self.vocab_size,
                             special_tokens=[self.unk_token])
        self.tokenizer.train_from_iterator(texts_iterator, trainer)
        return self  # allow chaining

    def save(self, path: str):
        """Save the trained tokenizer JSON to disk."""
        self.tokenizer.save(path)
        print(f"→ Saved tokenizer to: {path}")

    @classmethod
    def load(cls, path: str):
        """Load a previously saved tokenizer and wrap it in this class."""
        tok = Tokenizer.from_file(path)
        helper = cls()
        helper.tokenizer = tok
        print(f"→ Loaded tokenizer from: {path}")
        return helper

    def encode(self, text: str):
        """Encode a single string, returning the Encoding object."""
        return self.tokenizer.encode(text)

    def test(self, words: list[str]):
        """
        Encode a list of words or sentences and print out tokens/IDs.
        Demonstrates handling of seen vs. unseen inputs.
        """
        for w in words:
            enc = self.encode(w)
            print(f"\nInput: {w!r}")
            print(" Tokens:", enc.tokens)
            print("    IDs:", enc.ids)


if __name__ == "__main__":
    # 1. Load data
    ds = load_dataset("roneneldan/TinyStories")
    train_texts = ds["validation"]["text"]  # or 'train'

    # 2. Train & save
    tk = BPETokenizer(vocab_size=1000)
    tk.train(train_texts)
    tk.save("tinystories-bpe.json")

    # 3. Reload later (or in another script)
    tk2 = BPETokenizer.load("tinystories-bpe.json")

    # 4. Test on both seen and unseen words
    tk2.test([
        "slower",                          # likely in vocab → split into subwords
        "supercalifragilisticexpialidocious"  # totally unseen → [UNK]
    ])
