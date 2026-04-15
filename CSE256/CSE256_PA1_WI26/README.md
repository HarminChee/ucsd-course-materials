# CSE256 PA1 (WI26)  Harmin Qi

This repository contains implementations and experiments for:
- BOW baseline (2-layer / 3-layer MLP)
- Part 1: Word-level DAN
  - DAN with pretrained GloVe embeddings (50d)
  - DAN with randomly initialized embeddings (50d)
- Part 2: Subword DAN with BPE tokenization

All experiments can be run from `main.py` with a single command per model.

---

## Environment

- Python 3.9+ recommended
- PyTorch
- matplotlib

Example (optional) setup:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

---
Data

Expected files:
data/train.txt
data/dev.txt
data/glove.6B.50d-relativized.txt

---

How to Run
BOW baseline (generates BOW plots)

Single command:

python main.py --model BOW


Outputs:
train_accuracy.png
dev_accuracy.png

Common options:
--epochs (default: 20)
--lr (default: 1e-3)

---

Part 1a: DAN with GloVe (50d)

Single command:

python main.py --model DAN --glove_path data/glove.6B.50d-relativized.txt


Outputs:

dan_glove_acc.png

Common options:

--epochs (default: 20)

--lr (default: 1e-3)

--batch_size (default: 32)

--hidden_dim (default: 200)

--dropout (default: 0.3)

Part 1b: DAN with Random Embeddings (50d)

Single command:

python main.py --model DAN_RAND --emb_dim 50 --glove_path data/glove.6B.50d-relativized.txt


Outputs:

dan_rand_acc.png

Common options:

--epochs (default: 20)

--lr (default: 1e-3)

--batch_size (default: 32)

--hidden_dim (default: 200)

--dropout (default: 0.3)

--emb_dim (default: 50)

---

Part 2: SUBWORDDAN (BPE)

Single command (one run):

python main.py --model SUBWORDDAN --bpe_vocab_size 5000 --max_len 120 --emb_dim 50


Outputs:

subworddan_bpe<target>_acc.png (e.g., subworddan_bpe5000_acc.png)

Common options:

--bpe_vocab_size (default: 5000)

--max_len (default: 120)

--emb_dim (default: 50)

--epochs (default: 20)

--lr (default: 1e-3)

--batch_size (default: 32)

--hidden_dim (default: 200)

--dropout (default: 0.3)

To reproduce multiple BPE settings:

python main.py --model SUBWORDDAN --bpe_vocab_size 2000 --epochs 20 --lr 1e-3 --batch_size 32 --hidden_dim 200 --dropout 0.3 --max_len 120 --emb_dim 50
python main.py --model SUBWORDDAN --bpe_vocab_size 5000 --epochs 20 --lr 1e-3 --batch_size 32 --hidden_dim 200 --dropout 0.3 --max_len 120 --emb_dim 50
python main.py --model SUBWORDDAN --bpe_vocab_size 10000 --epochs 20 --lr 1e-3 --batch_size 32 --hidden_dim 200 --dropout 0.3 --max_len 120 --emb_dim 50

---

Example Results (from one reference run)
BOW
2 layers:
Epoch #5: train acc 0.834, dev acc 0.722
Epoch #10: train acc 0.952, dev acc 0.724
Epoch #15: train acc 0.980, dev acc 0.719
Epoch #20: train acc 0.984, dev acc 0.714
Best dev acc 0.735 at epoch 3

3 layers:
Epoch #5: train acc 0.919, dev acc 0.701
Epoch #10: train acc 0.981, dev acc 0.679
Epoch #15: train acc 0.979, dev acc 0.695
Epoch #20: train acc 0.988, dev acc 0.692
Best dev acc 0.728 at epoch 1

DAN (GloVe 50d)
Read in 14923 vectors of size 50
Epoch #5: train acc 0.918, dev acc 0.791
Epoch #10: train acc 0.988, dev acc 0.794
Epoch #15: train acc 0.998, dev acc 0.789
Epoch #20: train acc 1.000, dev acc 0.781
Best dev acc 0.794 at epoch 6

DAN_RAND (50d)
Read in 14923 vectors of size 50
Epoch #5: train acc 0.841, dev acc 0.739
Epoch #10: train acc 0.957, dev acc 0.757
Epoch #15: train acc 0.994, dev acc 0.753
Epoch #20: train acc 0.999, dev acc 0.755
Best dev acc 0.761 at epoch 16


SUBWORDDAN (BPE)
bpe_vocab_target=2000, actual_vocab=1790, best_dev=0.749, best_epoch=16
bpe_vocab_target=5000, actual_vocab=4558, best_dev=0.753, best_epoch=7
bpe_vocab_target=10000, actual_vocab=9305, best_dev=0.763, best_epoch=6


Generated plots:

dan_glove_acc.png

dan_rand_acc.png

subworddan_bpe2000_acc.png

subworddan_bpe5000_acc.png

subworddan_bpe10000_acc.png

train_accuracy.png

dev_accuracy.png