<div align="center">

# Handwriting Line Recognition (IAM) – Transformer CTC

Single–model implementation inspired by the paper:

**"A Multipurpose Vision Backbone for OCR Tasks"** – <https://arxiv.org/abs/2104.07787>

</div>

## 1. Overview

This repository contains a PyTorch implementation of an isometric convolutional + Transformer encoder architecture for handwritten text line recognition trained with CTC loss. The design follows the spirit of the backbone described in the referenced paper: a space‑to‑depth stem, a stack of lightweight (fused) inverted bottleneck blocks with constant spatial scale, and a Transformer encoder operating on the width dimension with relative position bias. Long lines are handled through overlapping width chunking to bound attention cost and preserve local context.

### Key Ideas
* **Isometric CNN Backbone** – Preserves width resolution; height reduced only once (space‑to‑depth + final collapse) for stable sequence alignment.
* **Fused Inverted Bottlenecks** – Expansion → depthwise‑like 3×3 (implemented as standard conv for simplicity) → projection with residual.
* **Space‑to‑Depth (4×)** – Converts (1,H,W) → (16,H/4,W/4) while keeping more local detail per token.
* **Height Collapse Block** – Single stride‑free convolution that reduces height to 1 and lifts channels to the Transformer dimension (256).
* **Transformer Encoder** – Multi‑layer self‑attention with learned 1‑D relative position bias (clipped) plus GELU MLP.
* **Overlapping Chunking** – Wide lines split into overlapping windows (`chunk_width`, `pad`) processed independently; central valid regions are concatenated then trimmed to `ceil(W/4)` tokens.
* **CTC Loss** – Alignment‑free training; blank index = 0.

## 2. Repository Structure
```
config.py                 # Hyperparameters & runtime options
scripts/train.py          # Training entry
dataset/iam_dataset.py    # IAM dataset parser & line image loader
dataset/transforms.py     # Data augmentation pipeline
models/model.py           # Complete model (backbone + Transformer + CTC head)
training/train_loop.py    # Training/eval loops (mixed precision, early stopping)
training/metrics.py       # CER / sequence & word accuracy / greedy decode
training/early_stopping.py
training/debug_helpers.py # Introspection utilities (entropy, distribution checks)
utils/label_converter.py  # Label ↔ index conversion (blank=0)
resources/                # Checkpoints & best model (created at runtime)
tests/test_shapes.py      # Shape consistency test (W → ⌈W/4⌉)
```

## 3. Installation
Python 3.10 recommended.
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If using GPU: verify `torch.cuda.is_available()` and install the appropriate CUDA wheel.

## 4. Data (IAM)
Expected layout:
```
data/iam_dataset/
	lines/<writer>/<form>/<line_id>.png
	xml/*.xml
```
1. Download IAM line images + XML annotations (follow IAM license terms).
2. Place all XML into `data/iam_dataset/xml/` and line images under the nested `lines/` structure.
3. (Optional) Limit parsed samples by setting `MAX_DATASET_SAMPLES` in `config.py` for quicker experiments.

The parser validates image existence and dimensions; missing or corrupt images are reported and skipped (dummy fallback used for rare failures).

## 5. Configuration Highlights (`config.py`)
| Setting | Purpose |
|---------|---------|
| `PAPER_MODEL.depth` | Transformer encoder depth (default reduced for warm start). |
| `BATCH_SIZE` | Lines per batch (variable width padded dynamically). |
| `chunk_width`, `pad` | Chunk window width & side overlap for long lines. |
| `EARLY_STOPPING` | CER‑based patience, min delta, mode. |
| `MIXED_PRECISION` | Enables `torch.cuda.amp` forward/backward. |
| `GRAD_ACCUM_STEPS` | Virtual batch size via gradient accumulation. |
| `FAST_DEV_RUN` | Limit batches/epoch for smoke tests. |
| `USE_LANGUAGE_MODEL` | Placeholder flag (beam/LM fusion not yet re‑added). |

## 6. Running Training
```bash
python scripts/train.py
```
Output per epoch (or dev batch window): loss, CER, sequence accuracy, word accuracy. Checkpoints appear under:
```
resources/checkpoints/
resources/best_model/best_model.pth
```
Early stopping (CER) restores best weights if enabled.

### Fast Dev Cycle
Set in `config.py`:
```
FAST_DEV_RUN = True
MAX_BATCHES_PER_EPOCH = 5
PAPER_MODEL["depth"] = 2
```
Then run one or two epochs to validate plumbing before scaling depth/epochs back up.

## 7. Decoding & Language Model
Current status: only greedy decoding (collapse repeats + remove blanks). The former beam + KenLM fusion harness was removed during simplification and will be reintroduced with a stable API:
Planned scoring = `log P_ctc + α log P_lm + β * length_norm + penalties(prior, transitions)`.

Rebuilding a KenLM 4‑gram (example):
```bash
python scripts/extract_corpus.py > corpus.txt
ctcdecode/third_party/kenlm/build/bin/lmplz -o 4 < corpus.txt > model.arpa
ctcdecode/third_party/kenlm/build/bin/build_binary model.arpa model.binary
```
Keep `model.binary` out of version control.

## 8. Metrics
* **CER** – Character error rate (Levenshtein / target length) %.
* **Sequence Accuracy** – Exact string matches %.
* **Word Accuracy** – Word overlap metric (simple presence test). 
* **Entropy / Blank Rate** – Printed in debug helper for the first batch.

Early stopping uses CER (lower better); best‑model selection currently uses validation loss by default (`EVAL_STRATEGY` configurable).

## 9. Architecture Details
```
Input (1, 40, W)
 → Pad per chunk (right) so each chunk width divisible by 4
 → SpaceToDepth(4): (16, 10, W/4)
 → Conv 3×3 (16→128), GELU
 → Conv 1×1 (128→64), GELU
 → 10 × Fused Inverted Bottleneck (64→512→64) + residual
 → ReduceConv (kernel (10,3)) → (256, 1, W/4)
 → Reshape → sequence (T≈⌈W/4⌉, 256)
 → Add sinusoidal pos encoding
 → N × TransformerEncoderLayer (MHSA + relative pos bias + MLP)
 → Linear → logits (T, vocab)
 → CTC Loss
```
Relative position bias: learned embedding of clipped distance added to attention logits per head.

Chunk merging trims overlapped feature tokens (`pad/4` on each internal side) ensuring assembled length does not exceed `ceil(W/4)`.

## 10. Performance Tips
| Tip | Effect |
|-----|--------|
| Lower initial depth (e.g. 4–6) | Faster stabilization before scaling up. |
| Mixed precision (`MIXED_PRECISION=True`) | 1.3–2× speedup on modern GPUs. |
| Gradient accumulation | Use when large width variability causes OOM. |
| Reduce `chunk_width` | Lowers attention quadratic cost; balance with more chunks. |
| Clip gradients (already on) | Stabilizes early CTC convergence. |

## 11. Roadmap
* Reintroduce beam search + LM fusion (α, β tuning + prior penalties).
* More rigorous chunk stitching (span mapping vs simple trim).
* Optional rotary or ALiBi positional alternatives for ablation.
* Batch‑level chunk processing to reduce Python overhead.
* Full validation vectorization for very large dev sets.
* Additional unit tests (LM fusion, chunk boundary invariants, decode idempotence).

## 12. Troubleshooting
| Symptom | Suggestion |
|---------|------------|
| CER flat, blanks low probability | Check label mapping & vocab size (blank must be index 0). |
| CER flat, blanks ~1.0 | Sequence too short: verify backbone height collapse & token length ≈ `ceil(W/4)`. |
| OOM on long lines | Decrease `chunk_width` or enable gradient accumulation. |
| Slow startup | Enable `FAST_DEV_RUN` until pipeline validated. |
| Width assertion in SpaceToDepth | Ensure chunk padding logic executed; avoid external random cropping that breaks divisibility. |

## 13. Citation
If you use this code or derivative ideas, please cite the original backbone paper:

```
@misc{diaz2021rethinkingtextlinerecognition,
      title={Rethinking Text Line Recognition Models}, 
      author={Daniel Hernandez Diaz and Siyang Qin and Reeve Ingle and Yasuhisa Fujii and Alessandro Bissacco},
      year={2021},
      eprint={2104.07787},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2104.07787}, 
}
```

Also cite IAM dataset per its license when publishing results.

## 14. License & Acknowledgements
Code in this repo is provided for research purposes; verify any third‑party components (e.g., KenLM, editdistance) under their respective licenses. The architectural inspiration comes from the referenced paper; any deviations (simplifications in bottlenecks, chunk trimming heuristic) are noted in comments.

---

Questions or improvement ideas? Feel free to open an issue or submit a PR.
