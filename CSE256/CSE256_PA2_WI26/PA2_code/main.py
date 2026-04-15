import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import TransformerEncoder, EncoderClassifier, TransformerDecoderLM
from utilities import Utilities


seed = 42
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 16
block_size = 32
learning_rate = 1e-3
n_embd = 64
n_head = 2
n_layer = 4

eval_interval = 100
max_iters = 500
eval_iters = 200

n_hidden = 100
n_output = 3
epochs_CLS = 15


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def load_texts(directory):
    texts = []
    files = os.listdir(directory)
    for filename in files:
        if "test" in filename:
            continue
        with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
            texts.append(f.read())
    return texts


def collate_batch(batch):
    data, labels = zip(*batch)
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]
    padded_sequences = torch.nn.functional.pad(
        padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0
    )
    labels = torch.stack(labels)
    return padded_sequences, labels


def compute_classifier_accuracy(classifier, data_loader):
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            logits = classifier(X)
            pred = torch.argmax(logits, dim=1)
            total_correct += (pred == Y).sum().item()
            total_samples += Y.size(0)
    classifier.train()
    return 100.0 * total_correct / total_samples


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    decoderLMmodel.eval()
    losses = []
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            loss = decoderLMmodel(X, Y)
            losses.append(loss.item())
            if len(losses) >= eval_iters:
                break
    decoderLMmodel.train()
    mean_loss = torch.tensor(losses).mean()
    return torch.exp(mean_loss).item()


def run_part1(tokenizer, train_CLS_loader, test_CLS_loader, out_dir):
    encoder = TransformerEncoder(
        vocab_size=tokenizer.vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
        ffn_hidden=100,
    ).to(device)

    classifier = EncoderClassifier(encoder, n_embd=n_embd, n_hidden=n_hidden, n_output=n_output).to(device)
    opt_cls = torch.optim.AdamW(classifier.parameters(), lr=learning_rate)

    print("Encoder params:", count_params(encoder))
    print("Training classifier ...")

    for epoch in range(epochs_CLS):
        total_loss = 0.0
        for xb, yb in train_CLS_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = classifier(xb)
            loss = F.cross_entropy(logits, yb)
            opt_cls.zero_grad(set_to_none=True)
            loss.backward()
            opt_cls.step()
            total_loss += loss.item()

        train_acc = compute_classifier_accuracy(classifier, train_CLS_loader)
        test_acc = compute_classifier_accuracy(classifier, test_CLS_loader)
        print(f"epoch {epoch+1}/{epochs_CLS} loss {total_loss/len(train_CLS_loader):.4f} train_acc {train_acc:.2f} test_acc {test_acc:.2f}")

    u = Utilities(tokenizer, encoder)
    u.sanity_check(
        "That is in Israel\\x27s interest , Palestine\\x27s interest , America\\x27s interest , and the world\\x27s interest .",
        block_size,
        out_prefix=os.path.join(out_dir, "attn_part1_sent1"),
        causal=False,
    )
    u.sanity_check(
        "We will not tire , we will not falter , and we will not fail .",
        block_size,
        out_prefix=os.path.join(out_dir, "attn_part1_sent2"),
        causal=False,
    )

    return encoder


def run_lm(tokenizer, train_LM_loader, window_size, out_dir, tag):
    decoder = TransformerDecoderLM(
        vocab_size=tokenizer.vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
        ffn_hidden=100,
        window_size=window_size,
    ).to(device)

    print("Decoder params:", count_params(decoder))
    opt_lm = torch.optim.AdamW(decoder.parameters(), lr=learning_rate)

    print("Training decoder LM ...")
    for i, (xb, yb) in enumerate(train_LM_loader):
        if i >= max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)
        loss = decoder(xb, yb)
        opt_lm.zero_grad(set_to_none=True)
        loss.backward()
        opt_lm.step()

        if (i + 1) % eval_interval == 0:
            ppl = compute_perplexity(decoder, train_LM_loader, eval_iters=50)
            print(f"iter {i+1}/{max_iters} loss {loss.item():.4f} train_ppl {ppl:.2f}")

    def load_lm(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    obama_text = load_lm("speechesdataset/test_LM_obama.txt")
    wbush_text = load_lm("speechesdataset/test_LM_wbush.txt")
    hbush_text = load_lm("speechesdataset/test_LM_hbush.txt")

    obama_ds = LanguageModelingDataset(tokenizer, obama_text, block_size)
    wbush_ds = LanguageModelingDataset(tokenizer, wbush_text, block_size)
    hbush_ds = LanguageModelingDataset(tokenizer, hbush_text, block_size)

    obama_loader = DataLoader(obama_ds, batch_size=batch_size, shuffle=False)
    wbush_loader = DataLoader(wbush_ds, batch_size=batch_size, shuffle=False)
    hbush_loader = DataLoader(hbush_ds, batch_size=batch_size, shuffle=False)

    ppl_obama = compute_perplexity(decoder, obama_loader, eval_iters=eval_iters)
    ppl_wbush = compute_perplexity(decoder, wbush_loader, eval_iters=eval_iters)
    ppl_hbush = compute_perplexity(decoder, hbush_loader, eval_iters=eval_iters)

    print(f"test_ppl_obama {ppl_obama:.2f}")
    print(f"test_ppl_wbush {ppl_wbush:.2f}")
    print(f"test_ppl_hbush {ppl_hbush:.2f}")

    u = Utilities(tokenizer, decoder)
    u.sanity_check(
        "That is in Israel\\x27s interest , Palestine\\x27s interest , America\\x27s interest , and the world\\x27s interest .",
        block_size,
        out_prefix=os.path.join(out_dir, f"attn_{tag}_sent1"),
        causal=True,
    )
    u.sanity_check(
        "We will not tire , we will not falter , and we will not fail .",
        block_size,
        out_prefix=os.path.join(out_dir, f"attn_{tag}_sent2"),
        causal=True,
    )

    return ppl_obama, ppl_wbush, ppl_hbush


def main(args):
    os.makedirs("artifacts/logs", exist_ok=True)
    os.makedirs("artifacts/attn", exist_ok=True)

    print("Loading data and creating tokenizer ...")
    texts = load_texts("speechesdataset")
    tokenizer = SimpleTokenizer(" ".join(texts))
    print("Vocabulary size is", tokenizer.vocab_size)

    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)

    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=False)

    with open("speechesdataset/train_LM.txt", "r", encoding="utf-8") as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText, block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

    if args.part == "part1":
        run_part1(tokenizer, train_CLS_loader, test_CLS_loader, out_dir="artifacts/attn")
        return

    if args.part == "part2":
        run_lm(tokenizer, train_LM_loader, window_size=None, out_dir="artifacts/attn", tag="part2")
        return

    if args.part == "part3":
        run_lm(tokenizer, train_LM_loader, window_size=args.local_window, out_dir="artifacts/attn", tag=f"part3_w{args.local_window}")
        return

    if args.part == "all":
        run_part1(tokenizer, train_CLS_loader, test_CLS_loader, out_dir="artifacts/attn")
        run_lm(tokenizer, train_LM_loader, window_size=None, out_dir="artifacts/attn", tag="part2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=str, default="all", choices=["all", "part1", "part2", "part3"])
    parser.add_argument("--local_window", type=int, default=8)
    args = parser.parse_args()
    main(args)
