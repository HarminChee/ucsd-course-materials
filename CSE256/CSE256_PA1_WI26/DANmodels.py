import torch
from torch import nn
from torch.utils.data import Dataset
from collections import Counter, defaultdict
from sentiment_data import read_sentiment_examples


class SentimentDatasetDAN(Dataset):
    def __init__(self, data_path, word_indexer, max_len=60):
        self.examples = read_sentiment_examples(data_path)
        self.word_indexer = word_indexer
        self.max_len = int(max_len)

        pad_id = self.word_indexer.index_of("PAD")
        unk_id = self.word_indexer.index_of("UNK")

        sentences = []
        labels = []

        for ex in self.examples:
            ids = []
            for w in ex.words[: self.max_len]:
                if self.word_indexer.contains(w):
                    ids.append(self.word_indexer.index_of(w))
                else:
                    ids.append(unk_id)

            while len(ids) < self.max_len:
                ids.append(pad_id)

            sentences.append(ids)
            labels.append(ex.label)

        self.sentences = torch.tensor(sentences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]


class BPETrainer:
    def __init__(self, vocab_size=5000, end_of_word="</w>"):
        self.vocab_size = int(vocab_size)
        self.end_of_word = end_of_word
        self.merges = []
        self.ranks = {}

    def _word_to_symbols(self, word):
        return tuple(list(word) + [self.end_of_word])

    def _get_stats(self, vocab):
        pairs = defaultdict(int)
        for symbols, freq in vocab.items():
            if len(symbols) < 2:
                continue
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_vocab(self, pair, vocab):
        a, b = pair
        merged_token = a + b
        new_vocab = {}
        for symbols, freq in vocab.items():
            new_syms = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_syms.append(merged_token)
                    i += 2
                else:
                    new_syms.append(symbols[i])
                    i += 1
            t = tuple(new_syms)
            new_vocab[t] = new_vocab.get(t, 0) + freq
        return new_vocab

    def fit(self, word_freq):
        vocab = {}
        for w, f in word_freq.items():
            if not w:
                continue
            syms = self._word_to_symbols(w)
            vocab[syms] = vocab.get(syms, 0) + int(f)

        symbols_set = set()
        for syms in vocab.keys():
            symbols_set.update(syms)

        while len(symbols_set) < self.vocab_size:
            pairs = self._get_stats(vocab)
            if not pairs:
                break
            best = max(pairs.items(), key=lambda x: x[1])[0]
            vocab = self._merge_vocab(best, vocab)
            self.merges.append(best)

            symbols_set = set()
            for syms in vocab.keys():
                symbols_set.update(syms)

        self.ranks = {m: i for i, m in enumerate(self.merges)}
        return self

    def build_tokenizer(self):
        return BPETokenizer(self.ranks, end_of_word=self.end_of_word)


class BPETokenizer:
    def __init__(self, ranks, end_of_word="</w>"):
        self.ranks = dict(ranks)
        self.end_of_word = end_of_word

    def _get_pairs(self, symbols):
        pairs = set()
        for i in range(len(symbols) - 1):
            pairs.add((symbols[i], symbols[i + 1]))
        return pairs

    def encode_word(self, word):
        if not word:
            return []
        symbols = tuple(list(word) + [self.end_of_word])

        while True:
            pairs = self._get_pairs(symbols)
            candidates = [(self.ranks[p], p) for p in pairs if p in self.ranks]
            if not candidates:
                break
            _, best = min(candidates, key=lambda x: x[0])
            a, b = best
            merged = a + b
            new_syms = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_syms.append(merged)
                    i += 2
                else:
                    new_syms.append(symbols[i])
                    i += 1
            symbols = tuple(new_syms)

        out = []
        for s in symbols:
            if s == self.end_of_word:
                continue
            if s.endswith(self.end_of_word):
                out.append(s[: -len(self.end_of_word)])
            else:
                out.append(s)
        return [t for t in out if t]

    def encode_sentence(self, words):
        pieces = []
        for w in words:
            pieces.extend(self.encode_word(w))
        return pieces


class SubwordVocab:
    def __init__(self):
        self.token_to_id = {"PAD": 0, "UNK": 1}
        self.id_to_token = ["PAD", "UNK"]

    def add(self, token):
        if token in self.token_to_id:
            return
        self.token_to_id[token] = len(self.id_to_token)
        self.id_to_token.append(token)

    def get_id(self, token):
        return self.token_to_id.get(token, 1)

    def __len__(self):
        return len(self.id_to_token)


class SentimentDatasetSubwordDAN(Dataset):
    def __init__(self, data_path, tokenizer, vocab, max_len=120):
        self.examples = read_sentiment_examples(data_path)
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_len = int(max_len)

        sentences = []
        labels = []

        for ex in self.examples:
            subwords = self.tokenizer.encode_sentence(ex.words)
            subwords = subwords[: self.max_len]
            ids = [self.vocab.get_id(t) for t in subwords]
            while len(ids) < self.max_len:
                ids.append(0)
            sentences.append(ids)
            labels.append(ex.label)

        self.sentences = torch.tensor(sentences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]


class DAN(nn.Module):
    def __init__(self, embedding_layer, emb_dim, hidden_dim, num_classes=2, dropout=0.3):
        super().__init__()
        self.embedding = embedding_layer
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        emb = self.embedding(x)
        mask = (x != 0).unsqueeze(-1)
        emb = emb * mask
        summed = emb.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        avg = summed / counts
        h = self.fc1(avg)
        h = self.relu(h)
        h = self.dropout(h)
        out = self.fc2(h)
        return self.log_softmax(out)


def build_bpe_and_vocab(train_path, vocab_size=5000, min_freq=1):
    train_examples = read_sentiment_examples(train_path)
    word_freq = Counter()
    for ex in train_examples:
        for w in ex.words:
            if w:
                word_freq[w] += 1

    trainer = BPETrainer(vocab_size=vocab_size).fit(word_freq)
    tokenizer = trainer.build_tokenizer()

    subword_freq = Counter()
    for w, f in word_freq.items():
        pieces = tokenizer.encode_word(w)
        for p in pieces:
            subword_freq[p] += f

    vocab = SubwordVocab()
    for token, f in subword_freq.most_common():
        if f >= min_freq:
            vocab.add(token)

    return tokenizer, vocab
