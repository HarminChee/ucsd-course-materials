import os
import numpy as np
import matplotlib.pyplot as plt
import torch


class Utilities:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def sanity_check(self, sentence, block_size, out_prefix="attention", causal=False):
        wordids = self.tokenizer.encode(sentence)
        L = min(len(wordids), block_size)
        padded = wordids[:block_size] + [0] * (block_size - len(wordids))
        input_tensor = torch.tensor(padded, dtype=torch.long).unsqueeze(0)

        try:
            device = next(self.model.parameters()).device
            input_tensor = input_tensor.to(device)
        except Exception:
            pass

        out = self.model(input_tensor)

        if isinstance(out, tuple) and len(out) == 2:
            _, attn_maps = out
        else:
            raise RuntimeError("Model output must be (output, attn_maps)")

        print("Input tensor shape:", input_tensor.shape)
        print("Number of attention maps:", len(attn_maps))

        os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

        for j, attn_map in enumerate(attn_maps):
            a = attn_map.squeeze(0).detach().float().cpu().numpy()

            if a.ndim != 2:
                raise RuntimeError(f"Expected attn_map to be 2D after squeeze, got shape={a.shape}")

            a = a[:L, :L]

            row_sums = torch.sum(attn_map[0, :L, :L], dim=1).detach().cpu().numpy()
            if np.any(row_sums < 0.99) or np.any(row_sums > 1.01):
                print("Failed normalization test: probabilities do not sum to 1.0 over rows")
                print("Row sums:", row_sums)

            if causal:
                m = np.triu(np.ones((L, L), dtype=bool), k=1)
                a = a.copy()
                a[m] = np.nan

            cmap = plt.cm.hot.copy()
            cmap.set_bad(color="white")

            fig, ax = plt.subplots()
            im = ax.imshow(a, cmap=cmap, interpolation="nearest")
            ax.xaxis.tick_top()
            fig.colorbar(im, ax=ax)
            ax.set_title(f"Attention Map {j + 1} (L={L})")

            out_path = f"{out_prefix}_layer{j+1}.png"
            fig.savefig(out_path, bbox_inches="tight", dpi=200)
            plt.close(fig)
