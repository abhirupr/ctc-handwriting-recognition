git clone https://github.com/yourname/ctc-handwriting-rtl-r.git
## Handwriting Recognition with CTC (IAM Dataset)

Lightweight CNN + BiLSTM + CTC handwriting line recognizer with: 

* Greedy decoding (working) 
* Beam search + KenLM interface (scaffolded; LM optional) 
* Early stopping (CER-based) & checkpoint rotation 
* Character Error Rate (CER), sequence and word accuracy metrics 
* Optional data augmentation (rotation, shear, noise) 

Status: Stable greedy CTC training. Transformer / advanced CNN backbone classes in `models/rtlr_model.py` are experimental and currently unused by the active `CTCRecognitionModel`.

---

### 1. Environment

Python 3.10 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you accidentally created a 3.12 venv and need 3.10, recreate it with the correct interpreter (e.g. `python3.10 -m venv .venv`).

---

### 2. Dataset (IAM)

Expected structure (root at `data/iam_dataset/`):
```
data/iam_dataset/
	lines/<writer>/<form>/<line_id>.png
	xml/*.xml
```
Place the extracted IAM line images under `lines/` and XML descriptors in `xml/`.

The loader will parse XML until it collects up to 1000 samples (debug limit you can raise in `dataset/iam_dataset.py`). Adjust or remove the early stop if you want the full dataset.

---

### 3. Training

Configuration lives in `config.py`.

Key knobs:
* EPOCHS, BATCH_SIZE, LEARNING_RATE
* EARLY_STOPPING (CER based, mode 'min')
* DATA_AUGMENTATION toggles
* USE_LANGUAGE_MODEL (currently only affects planned beam search path)

Run training:
```bash
python scripts/train.py
```

During training you'll see per‑epoch: train loss/CER, validation loss/CER, sequence & word accuracy (val). Checkpoints rotate in `resources/checkpoints/` and best model stored in `resources/best_model/`.

---

### 4. Language Model (KenLM)

Already built artifacts (not tracked): `model.arpa`, `model.binary`, and `corpus.txt` (extracted text lines). These are ignored via `.gitignore`.

To rebuild (after extracting corpus):
```bash
# Build corpus (if needed)
python scripts/extract_corpus.py > /dev/null 2>&1  # or run directly

# KenLM (example)
ctcdecode/third_party/kenlm/build/bin/lmplz -o 4 < corpus.txt > model.arpa
ctcdecode/third_party/kenlm/build/bin/build_binary model.arpa model.binary
```

Beam + LM decoding wrapper scaffolds are in `models/rtlr_model.py` (`PyCTCBeamDecoder`, `CTCDecoder`). They currently fall back to greedy if `ctcdecode` isn't installed or LM path absent. Integrate at evaluation time by replacing `greedy_decode` calls with the decoder interface.

---

### 5. Metrics

Reported:
* CER (character error rate, %)
* Sequence accuracy (exact match %)
* Word accuracy (%)

Early stopping monitors CER (lower better). Validation selection for best model currently uses loss unless you switch `EVAL_STRATEGY` in `config.py`.

---

### 6. Code Layout
```
config.py                # Global hyperparameters & paths
scripts/train.py         # Training entry point
dataset/iam_dataset.py   # IAM dataset + augmentation hook
dataset/transforms.py    # Lightweight augmentation
models/rtlr_model.py     # Active CNN+BiLSTM CTC model + experimental blocks
training/train_loop.py   # Train/eval loops, metrics integration
training/metrics.py      # CER, accuracy, greedy decode
training/early_stopping.py
training/debug_helpers.py
utils/label_converter.py # Label/index conversions (blank=0)
resources/               # (Generated) checkpoints & best model
```

Experimental / unused (safe to prune if not needed):
* `PositionalEncoding`, `TransformerEncoderBlock`, `SelfAttentionEncoder`, `CNNBackbone` in `models/rtlr_model.py`.

---

### 7. Removing / Ignoring Large Artifacts

`.gitignore` already excludes: checkpoints, kenlm, data, LM binaries, corpus. Do not commit `model.binary`, `model.arpa`, or dataset images.

---

### 8. Next Steps (Planned Improvements)
* Wire beam search + KenLM into evaluation
* Expand to full IAM (remove 1000 sample limit)
* Stronger augmentation (elastic distortions, brightness/contrast currently stubbed)
* Label smoothing or alternative loss variations
* Transformer encoder re‑integration (if performance gains justify complexity)

---

### 9. Troubleshooting
* CER stuck high & blank probability ~1.0: ensure time dimension > target length (check CNN pooling).
* Early stopping triggers too soon: increase `patience` or raise `min_delta` sensitivity.
* LM not used: confirm `ctcdecode` installed and `model.binary` path exists.

---

### 10. Performance Snapshot (Example Run)
On ~1000 debug samples subset: train CER ~8–9%, validation CER ~22–24% (greedy, no LM). Expect higher CER on small subset; full dataset & beam+LM should reduce further.

---

### 11. License
MIT. See `LICENSE` (if present) or add one if missing.

---

### 12. Citation
If you use this repo academically, please cite relevant CTC / IAM resources.

---

Feel free to streamline by deleting experimental blocks if you want only the minimal CNN+BiLSTM pipeline.
