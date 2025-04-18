from transformers import AutoTokenizer

# This should now import cleanly:
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# And this should print subword splits without errors:
print(tokenizer.tokenize("slower"))
