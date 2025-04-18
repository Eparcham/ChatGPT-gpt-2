# ChatGPT-gpt-2-
# NLP & PyTorch Examples

This repository provides self-contained code snippets demonstrating a variety of NLP tasks using PyTorch and popular tokenization libraries. Each section lives in a single file and can be run independently.

## Sections

1. **TinyStories Dataset Loader**  
   Load the `roneneldan/TinyStories` dataset via 🤗 Datasets.  
2. **Text to Vectors**  
   Convert raw text into embeddings using a Hugging Face tokenizer + `nn.Embedding`.  
3. **Main Token Types**  
   Examples of word-level and character-level token ID mappings.  
4. **Byte Pair Encoding (BPE)**  
   Train a BPE tokenizer on a custom corpus and generate subword embeddings.  
5. **English NLP Libraries**  
   Quick demos for NLTK, spaCy, and Gensim → PyTorch integration.  
6. **tiktoken (OpenAI)**  
   Encode text for GPT-4 and map token IDs to embeddings.  
7. **Persian Tokenizers**  
   Tokenize Persian text with Hazm and embed tokens.  
8. **Handling Unknown Tokens**  
   Demonstrate how the Hugging Face tokenizer handles unseen words.

---

## Prerequisites

- Python 3.7 or higher  
- [PyTorch](https://pytorch.org/) 1.7+  
- [Transformers](https://github.com/huggingface/transformers)  
- [Datasets](https://github.com/huggingface/datasets)  
- [tokenizers](https://github.com/huggingface/tokenizers)  
- `nltk`, `spacy`, `gensim`, `tiktoken`, `hazm`

Install all dependencies with:

```bash
pip install torch transformers datasets tokenizers nltk spacy gensim tiktoken hazm
python -m spacy download en_core_web_sm
