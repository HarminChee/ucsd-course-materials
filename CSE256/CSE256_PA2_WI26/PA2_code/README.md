# CSE 256 WI 2026 — PA2 Transformer Blocks

# Done by Harmin Qi at 14/2/2026 to fullfill CSE 256 PA2 Requirement

This codebase implements:
- A Transformer Encoder trained end-to-end for 3-way speech segment classification.
- A GPT-style Transformer Decoder Language Model (causal self-attention), plus an architectural variant with local-window causal attention.

No torch.nn.Transformer modules are used.

## Requirements

- Python
- PyTorch
- nltk
- matplotlib

## Data

All inputs are under `speechesdataset/`:
- Classification: `train_CLS.tsv`, `test_CLS.tsv`
- Language Modeling: `train_LM.txt`, `test_LM_obama.txt`, `test_LM_wbush.txt`, `test_LM_hbush.txt`

## Outputs

Report-ready artifacts are written to:
- `artifacts/logs/` (training logs)
- `artifacts/attn/` (attention heatmaps)

Decoder attention heatmaps mask the future region (causal mask) as white/transparent and plot only the real token length (no padding region).

## Run

From the `PA2_code` directory:

### Part 1 — Encoder + Classifier
Trains for 15 epochs and saves encoder attention maps.

```bash
python main.py --part part1 | tee artifacts/logs/part1.txt
```

Expected artifacts:
- `artifacts/logs/part1.txt`
- `artifacts/attn/attn_part1_sent1_layer*.png`
- `artifacts/attn/attn_part1_sent2_layer*.png`

### Part 2 — Decoder LM Baseline (Full causal attention)
Trains for 500 iterations and reports test perplexities.

```bash
python main.py --part part2 | tee artifacts/logs/part2_baseline.txt
```

Expected artifacts:
- `artifacts/logs/part2_baseline.txt`
- `artifacts/attn/attn_part2_sent1_layer*.png`
- `artifacts/attn/attn_part2_sent2_layer*.png`

### Part 3 — Local-window causal attention
Runs the decoder with a limited causal attention window size.

```bash
python main.py --part part3 --local_window 8  | tee artifacts/logs/part3_w8.txt
python main.py --part part3 --local_window 16 | tee artifacts/logs/part3_w16.txt
```

Expected artifacts:
- `artifacts/logs/part3_w8.txt`, `artifacts/logs/part3_w16.txt`
- `artifacts/attn/attn_part3_w8_sent*_layer*.png`
- `artifacts/attn/attn_part3_w16_sent*_layer*.png`

## Notes

- If you want to re-generate all artifacts from scratch:
```bash
rm -rf artifacts/logs artifacts/attn
mkdir -p artifacts/logs artifacts/attn
python main.py --part part1 | tee artifacts/logs/part1.txt
python main.py --part part2 | tee artifacts/logs/part2_baseline.txt
python main.py --part part3 --local_window 8  | tee artifacts/logs/part3_w8.txt
python main.py --part part3 --local_window 16 | tee artifacts/logs/part3_w16.txt
```
