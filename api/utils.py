def clean_sentence(output, idx2word):
    """Converts the model's output into a human-readable sentence."""
    words = [idx2word[idx] for idx in output if idx2word[idx] != "<end>"]
    return " ".join(words)
