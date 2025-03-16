import os
import pickle

""" vocab_path = os.path.join(os.getcwd(), 'vocab.pkl')
print(f"Checking for vocab.pkl at {vocab_path}")

if os.path.exists(vocab_path):
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    print(f"Vocabulary loaded successfully: {len(vocab)} words")
else:
    print("vocab.pkl not found!") """

import pickle

vocab_path = os.path.join(os.getcwd(), 'vocab.pkl')
print(f"Checking for vocab.pkl at {vocab_path}")

try:
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f, encoding='latin1')  # Add encoding option
    print(f"Vocabulary loaded successfully: {len(vocab)} words")
except ModuleNotFoundError as e:
    print(f"Error: {e}")

