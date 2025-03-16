from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS

import torch
from PIL import Image
import torchvision.transforms as transforms
import os

from models.architectures import load_encoder, load_decoder

import pickle

# Load the vocabulary from the vocab.pkl file
import pickle
import os

import pickle

import pickle
import os

import pickle

def load_vocab():
    try:
        vocab_path = os.path.join(os.getcwd(), 'vocab.pkl')  # Use absolute path for safety
        print(f"Attempting to load vocabulary from {vocab_path}")
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        print(f"Vocabulary loaded successfully: {len(vocab)} words")
        return vocab
    except Exception as e:
        print(f"Error loading vocabulary: {e}")
        return None






# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all origins (use specific origins in production)
CORS(app)

# Device setup (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
EMBED_SIZE = 256
HIDDEN_SIZE = 512
VOCAB_SIZE = 11543
NUM_LAYERS = 1

encoder = load_encoder(embed_size=EMBED_SIZE).to(device)
decoder = load_decoder(embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE, vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS).to(device)

encoder.eval()
decoder.eval()

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Helper function to generate captions
""" def generate_caption(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            features = encoder(image_tensor).unsqueeze(1)
            output = decoder.sample(features)

        sentence = " ".join([word for word in output if word not in ("<start>", "<end>", "<pad>")])
        return sentence

    except Exception as e:
        print(f"Error during prediction: {e}")
        return None """

""" def generate_caption(image_path):
    try:
        # Step 1: Load the image
        print(f"Loading image from: {image_path}")
        image = Image.open(image_path).convert("RGB")
        print("Image loaded successfully")

        # Step 2: Transform the image to tensor
        image_tensor = transform(image).unsqueeze(0).to(device)
        print(f"Image tensor shape: {image_tensor.shape}")  # Check the shape of the tensor

        # Step 3: Pass image tensor through encoder to extract features
        with torch.no_grad():
            features = encoder(image_tensor).unsqueeze(1)
            print(f"Features shape: {features.shape}")  # Check the shape of the features

            # Step 4: Pass features through the decoder to generate output
            output = decoder.sample(features)
            print(f"Decoder output: {output}")  # Check the raw output from the decoder

        # Step 5: Process the output to generate the final caption
        sentence = " ".join([word for word in output if word not in ("<start>", "<end>", "<pad>")])
        print(f"Generated caption: {sentence}")  # Final caption output

        return sentence

    except Exception as e:
        print(f"Error during prediction: {e}")
        return None """

def generate_caption(image_path):
    try:
        # Load image
        print(f"Loading image from: {image_path}")
        image = Image.open(image_path).convert("RGB")
        print("Image loaded successfully")

        # Preprocess image and extract features
        image_tensor = transform(image).unsqueeze(0).to(device)
        print(f"Image tensor shape: {image_tensor.shape}")

        with torch.no_grad():
            features = encoder(image_tensor).unsqueeze(1)
            print(f"Features shape: {features.shape}")
            output = decoder.sample(features)
            print(f"Decoder output (token IDs): {output}")

        # Load the vocabulary
        vocab = load_vocab()
        if not vocab:
            print("Vocabulary not loaded properly.")
            return None

        # Convert token IDs to words
        sentence = " ".join([vocab.idx2word.get(token, "<unk>") for token in output if token not in (1, 2, 3)])
        print(f"Generated sentence: {sentence}")
        return sentence

    except Exception as e:
        print(f"Error during prediction: {e}")
        return None




# API endpoint for generating captions
@app.route('/generate-caption', methods=['POST'])
def generate_caption_endpoint():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files['image']

    if not image.filename.lower().endswith(('png', 'jpg', 'jpeg')):
        return jsonify({"error": "Invalid image format. Only PNG, JPG, or JPEG are allowed."}), 400

    image_path = os.path.join("temp", image.filename)
    os.makedirs("temp", exist_ok=True)
    image.save(image_path)

    caption = generate_caption(image_path)

    try:
        os.remove(image_path)
    except Exception as e:
        print(f"Error removing temporary image: {e}")

    if caption:
        return jsonify({"caption": caption}), 200
    else:
        return jsonify({"error": "Failed to generate caption"}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
