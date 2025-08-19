import os, sys
import torch

# Ensure project root (parent of this tests directory) is on sys.path when run as a script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.model import CTCTransformerModel


def test_sequence_length_mapping():
    model = CTCTransformerModel(vocab_size=100, depth=2, chunk_width=320, pad=32)
    model.eval()
    # widths to test (not all divisible by 4 to exercise padding)
    widths = [160, 319, 320, 640, 725]
    with torch.no_grad():
        for w in widths:
            img = torch.randn(1, 1, 40, w)
            logits = model(img)  # (1,T,V)
            T = logits.size(1)
            expected = (w + 3) // 4  # ceil(w/4)
            # Allow small deviation (due to chunk trimming) but should not exceed +1
            assert abs(T - expected) <= 1, f"Width {w}: got T={T}, expected≈{expected}"  

if __name__ == "__main__":
    test_sequence_length_mapping()
    print("shape tests passed")
