from nltk.tokenize import TreebankWordTokenizer


class SimpleTokenizer:
    def __init__(self, text):
        self.tok = TreebankWordTokenizer()
        self.vocab = set()
        self.stoi = {}
        self.itos = {}
        self.build_vocab(text)

    def build_vocab(self, text):
        tokens = self.tok.tokenize(text)
        self.vocab = set(tokens)
        self.vocab_size = len(self.vocab) + 2
        self.stoi = {word: i for i, word in enumerate(self.vocab, start=2)}
        self.stoi["<pad>"] = 0
        self.stoi["<unk>"] = 1
        self.itos = {i: w for w, i in self.stoi.items()}

    def encode(self, text):
        tokens = self.tok.tokenize(text)
        return [self.stoi.get(w, self.stoi["<unk>"]) for w in tokens]

    def decode(self, indices):
        return " ".join([self.itos.get(i, "<unk>") for i in indices])
