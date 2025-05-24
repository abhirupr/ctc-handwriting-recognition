import torch

class LabelConverter:
    def __init__(self, vocab):
        # Blank token is at index 0
        self.vocab = ['<BLANK>'] + list(vocab)
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        
        print(f"🔤 Label Converter initialized:")
        print(f"   Vocab size: {len(self.vocab)} (including blank)")
        print(f"   Blank token: '{self.vocab[0]}' (index: 0)")
        print(f"   Sample chars: {self.vocab[1:11]}")
    
    def encode(self, texts):
        """Convert texts to character indices (excluding blank)"""
        encoded_texts = []
        for text in texts:
            indices = []
            for char in text:
                if char in self.char_to_idx:
                    indices.append(self.char_to_idx[char])
                else:
                    # Skip unknown characters with warning
                    print(f"⚠️ Unknown character: '{char}' (ord: {ord(char)}) - skipping")
            encoded_texts.append(torch.tensor(indices, dtype=torch.long))
        return encoded_texts
    
    def decode(self, indices_list):
        """Convert indices back to texts"""
        decoded_texts = []
        for indices in indices_list:
            chars = []
            for idx in indices:
                if isinstance(idx, torch.Tensor):
                    idx = idx.item()
                if 0 <= idx < len(self.vocab):
                    char = self.idx_to_char[idx]
                    if char != '<BLANK>':  # Skip blank tokens
                        chars.append(char)
                else:
                    print(f"⚠️ Invalid index: {idx} (vocab size: {len(self.vocab)})")
            decoded_texts.append(''.join(chars))
        return decoded_texts
