import torch

class LabelConverter:
    def __init__(self, vocab):
        self.vocab = vocab
        self.char2idx = {char: idx + 1 for idx, char in enumerate(vocab)}  # 0 is reserved for CTC blank
        self.idx2char = {idx + 1: char for idx, char in enumerate(vocab)}
        self.blank_idx = 0

    def encode(self, texts):
        """
        Args:
            texts (list of str): list of transcripts
        Returns:
            list of torch.Tensor: encoded labels
        """
        return [torch.tensor([self.char2idx[char] for char in text if char in self.char2idx], dtype=torch.long) for text in texts]

    def decode(self, indices, remove_repeats=True):
        """
        Decodes output indices from model predictions
        """
        decoded = []
        prev = self.blank_idx
        for idx in indices:
            if idx != self.blank_idx and (not remove_repeats or idx != prev):
                decoded.append(self.idx2char.get(idx, ''))
            prev = idx
        return ''.join(decoded)
