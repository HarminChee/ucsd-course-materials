import argparse
import time
import torch
from torch import nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from sentiment_data import read_word_embeddings
from BOWmodels import SentimentDatasetBOW, NN2BOW, NN3BOW
from DANmodels import SentimentDatasetDAN, DAN, SentimentDatasetSubwordDAN, build_bpe_and_vocab


def train_epoch(data_loader, model, loss_fn, optimizer, device):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()

    total_loss = 0.0
    correct = 0.0

    for X, y in data_loader:
        X = X.to(device)
        y = y.to(device)

        if X.dtype != torch.long:
            X = X.float()

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    return correct / size, total_loss / num_batches


def eval_epoch(data_loader, model, loss_fn, device):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()

    total_loss = 0.0
    correct = 0.0

    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(device)
            y = y.to(device)

            if X.dtype != torch.long:
                X = X.float()

            pred = model(X)
            loss = loss_fn(pred, y)

            total_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    return correct / size, total_loss / num_batches


def experiment(model, train_loader, dev_loader, epochs, lr, print_every, device):
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_accs = []
    dev_accs = []

    best_dev = -1.0
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        train_acc, _ = train_epoch(train_loader, model, loss_fn, optimizer, device)
        dev_acc, _ = eval_epoch(dev_loader, model, loss_fn, device)

        train_accs.append(train_acc)
        dev_accs.append(dev_acc)

        if dev_acc > best_dev:
            best_dev = dev_acc
            best_epoch = epoch

        if epoch % print_every == 0:
            print(f"Epoch #{epoch}: train acc {train_acc:.3f}, dev acc {dev_acc:.3f}")

    print(f"Best dev acc {best_dev:.3f} at epoch {best_epoch}")
    return train_accs, dev_accs, best_dev, best_epoch


def run_bow(epochs, lr, batch_size, hidden_size, device):
    start_time = time.time()

    train_data = SentimentDatasetBOW("data/train.txt")
    dev_data = SentimentDatasetBOW("data/dev.txt", vectorizer=train_data.vectorizer, train=False)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=False)

    print(f"Data loaded in : {time.time() - start_time} seconds")

    print("\n2 layers:")
    model2 = NN2BOW(input_size=512, hidden_size=hidden_size).to(device)
    experiment(model2, train_loader, dev_loader, epochs=epochs, lr=lr, print_every=5, device=device)

    print("\n3 layers:")
    model3 = NN3BOW(input_size=512, hidden_size=hidden_size).to(device)
    experiment(model3, train_loader, dev_loader, epochs=epochs, lr=lr, print_every=5, device=device)


def build_dan_loaders(word_indexer, max_len, batch_size):
    train_data = SentimentDatasetDAN("data/train.txt", word_indexer=word_indexer, max_len=max_len)
    dev_data = SentimentDatasetDAN("data/dev.txt", word_indexer=word_indexer, max_len=max_len)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=False)
    return train_loader, dev_loader


def run_dan_glove(epochs, lr, max_len, batch_size, hidden_dim, dropout, glove_path, device):
    embs = read_word_embeddings(glove_path)
    embedding_layer = embs.get_initialized_embedding_layer(frozen=False)

    train_loader, dev_loader = build_dan_loaders(embs.word_indexer, max_len=max_len, batch_size=batch_size)

    model = DAN(
        embedding_layer=embedding_layer,
        emb_dim=embedding_layer.embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=2,
        dropout=dropout
    ).to(device)

    train_accs, dev_accs, _, _ = experiment(model, train_loader, dev_loader, epochs=epochs, lr=lr, print_every=5, device=device)

    plt.figure(figsize=(8, 6))
    plt.plot(train_accs, label="DAN(GloVe)-train")
    plt.plot(dev_accs, label="DAN(GloVe)-dev")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"DAN with GloVe ({embedding_layer.embedding_dim}d)")
    plt.legend()
    plt.grid()
    plt.savefig("dan_glove_acc.png")


def run_dan_rand(epochs, lr, max_len, batch_size, hidden_dim, dropout, glove_path, emb_dim, device):
    embs = read_word_embeddings(glove_path)
    vocab_size = len(embs.word_indexer)

    if emb_dim <= 0:
        emb_dim = embs.get_embedding_length()

    embedding_layer = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

    train_loader, dev_loader = build_dan_loaders(embs.word_indexer, max_len=max_len, batch_size=batch_size)

    model = DAN(
        embedding_layer=embedding_layer,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        num_classes=2,
        dropout=dropout
    ).to(device)

    train_accs, dev_accs, _, _ = experiment(model, train_loader, dev_loader, epochs=epochs, lr=lr, print_every=5, device=device)

    plt.figure(figsize=(8, 6))
    plt.plot(train_accs, label="DAN(Rand)-train")
    plt.plot(dev_accs, label="DAN(Rand)-dev")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"DAN with Random Embeddings ({emb_dim}d)")
    plt.legend()
    plt.grid()
    plt.savefig("dan_rand_acc.png")


def run_subworddan(epochs, lr, bpe_vocab_size, max_len, batch_size, hidden_dim, dropout, emb_dim, device):
    tokenizer, vocab = build_bpe_and_vocab("data/train.txt", vocab_size=bpe_vocab_size, min_freq=1)
    vocab_size_actual = len(vocab)
    print(f"Built BPE vocab size: {vocab_size_actual}")

    train_data = SentimentDatasetSubwordDAN("data/train.txt", tokenizer=tokenizer, vocab=vocab, max_len=max_len)
    dev_data = SentimentDatasetSubwordDAN("data/dev.txt", tokenizer=tokenizer, vocab=vocab, max_len=max_len)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=False)

    embedding_layer = nn.Embedding(vocab_size_actual, emb_dim, padding_idx=0)

    model = DAN(
        embedding_layer=embedding_layer,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        num_classes=2,
        dropout=dropout
    ).to(device)

    train_accs, dev_accs, best_dev, best_epoch = experiment(
        model, train_loader, dev_loader, epochs=epochs, lr=lr, print_every=5, device=device
    )

    plt.figure(figsize=(8, 6))
    plt.plot(train_accs, label=f"SUBWORDDAN-train(v={bpe_vocab_size})")
    plt.plot(dev_accs, label=f"SUBWORDDAN-dev(v={bpe_vocab_size})")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"SUBWORDDAN (BPE vocab target={bpe_vocab_size}, actual={vocab_size_actual})")
    plt.legend()
    plt.grid()
    plt.savefig(f"subworddan_bpe{bpe_vocab_size}_acc.png")
    print(f"Saved subworddan_bpe{bpe_vocab_size}_acc.png")
    print(f"SUBWORDDAN summary: bpe_vocab_target={bpe_vocab_size}, actual_vocab={vocab_size_actual}, best_dev={best_dev:.3f}, best_epoch={best_epoch}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["BOW", "DAN", "DAN_RAND", "SUBWORDDAN"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_len", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=200)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--bpe_vocab_size", type=int, default=5000)
    parser.add_argument("--emb_dim", type=int, default=50)
    parser.add_argument("--glove_path", type=str, default="data/glove.6B.50d-relativized.txt")
    parser.add_argument("--bow_batch_size", type=int, default=16)
    parser.add_argument("--bow_hidden_size", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "BOW":
        run_bow(args.epochs, args.lr, args.bow_batch_size, args.bow_hidden_size, device)
    elif args.model == "DAN":
        run_dan_glove(args.epochs, args.lr, max_len=60, batch_size=args.batch_size, hidden_dim=args.hidden_dim, dropout=args.dropout, glove_path=args.glove_path, device=device)
    elif args.model == "DAN_RAND":
        run_dan_rand(args.epochs, args.lr, max_len=60, batch_size=args.batch_size, hidden_dim=args.hidden_dim, dropout=args.dropout, glove_path=args.glove_path, emb_dim=args.emb_dim, device=device)
    elif args.model == "SUBWORDDAN":
        run_subworddan(args.epochs, args.lr, args.bpe_vocab_size, args.max_len, args.batch_size, args.hidden_dim, args.dropout, args.emb_dim, device)
    else:
        raise ValueError("Unknown model type")


if __name__ == "__main__":
    main()
