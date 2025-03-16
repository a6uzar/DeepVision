import os
import torch
from models.architectures import EncoderCNN, DecoderRNN

MODEL_DIR = os.path.dirname(__file__)
ENCODER_MODEL_PATH = os.path.join(MODEL_DIR, "encoder-3.pkl")
DECODER_MODEL_PATH = os.path.join(MODEL_DIR, "decoder-3.pkl")

# Update the parameters based on your model's training configuration
EMBED_SIZE = 256  # Replace with the embedding size used during training
HIDDEN_SIZE = 512  # Replace with the hidden size used during training
VOCAB_SIZE = 5000  # Replace with your vocabulary size

def load_encoder():
    try:
        encoder = EncoderCNN(embed_size=EMBED_SIZE)
        encoder.load_state_dict(torch.load(ENCODER_MODEL_PATH, map_location=torch.device("cpu")))
        encoder.eval()  # Set to evaluation mode
        return encoder
    except Exception as e:
        print(f"Error loading encoder model: {e}")
        return None

def load_decoder():
    try:
        decoder = DecoderRNN(embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE, vocab_size=VOCAB_SIZE)
        decoder.load_state_dict(torch.load(DECODER_MODEL_PATH, map_location=torch.device("cpu")))
        decoder.eval()  # Set to evaluation mode
        return decoder
    except Exception as e:
        print(f"Error loading decoder model: {e}")
        return None
