# 🖋️ CTC-Handwriting-RTL-R

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Model](https://img.shields.io/badge/model-CTC--Transformer--Hybrid-green)]()
[![Paper](https://img.shields.io/badge/paper-Rethinking_Text_Line_Recognition_Models-blue)]()

Implementation of the paper **"Rethinking Text Line Recognition Models"** using a CNN + Transformer Encoder + CTC decoder (with optional Language Model integration). Supports training and evaluation on IAM and RIMES handwriting datasets.

---

## 🧠 Highlights

- ✅ Space-to-depth CNN backbone
- ✅ 16-layer Transformer encoder
- ✅ CTC decoder with optional KenLM integration (via `ctcdecode`)
- ✅ Fully scriptable training & inference
- ✅ IAM & RIMES dataset support

---

## ⚙️ Setup

```bash
# Clone repo
git clone https://github.com/yourname/ctc-handwriting-rtl-r.git
cd ctc-handwriting-rtl-r

# Create environment (optional)
python -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
