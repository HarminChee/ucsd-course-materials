import argparse 
import gzip
import math
import random
from collections import defaultdict
import copy

import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.nn.functional as F


# Haomin Qi
# Utility functions

def readGz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            yield eval(line)


def readCSV(path):
    f = gzip.open(path, 'rt', encoding='utf-8')
    f.readline()
    for line in f:
        yield line.strip().split(',')


def set_global_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Rating prediction

class MFWithBias:
    def __init__(
        self,
        n_factors=40,
        n_epochs=60,
        lr=0.004,
        reg=0.04,
        val_ratio=0.1,
        lr_decay_steps=(30, 45),
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.val_ratio = val_ratio
        self.lr_decay_steps = lr_decay_steps

    def fit(self, interactions_path):
        set_global_seeds(0)

        user_index = {}
        item_index = {}
        triplets = []
        all_ratings = []

        for u, b, r in readCSV(interactions_path):
            r = float(r)
            all_ratings.append(r)
            if u not in user_index:
                user_index[u] = len(user_index)
            if b not in item_index:
                item_index[b] = len(item_index)
            triplets.append((user_index[u], item_index[b], r, u, b))

        self.user_index = user_index
        self.item_index = item_index

        n_users = len(user_index)
        n_items = len(item_index)

        self.mu = float(sum(all_ratings) / len(all_ratings))

        k = self.n_factors
        rng = np.random.RandomState(0)
        self.P = 0.1 * rng.randn(n_users, k).astype(np.float32)
        self.Q = 0.1 * rng.randn(n_items, k).astype(np.float32)
        self.b_u = np.zeros(n_users, dtype=np.float32)
        self.b_i = np.zeros(n_items, dtype=np.float32)

        user_sum = defaultdict(float)
        user_cnt = defaultdict(int)
        item_sum = defaultdict(float)
        item_cnt = defaultdict(int)
        for _, _, r, u_id, b_id in triplets:
            user_sum[u_id] += r
            user_cnt[u_id] += 1
            item_sum[b_id] += r
            item_cnt[b_id] += 1

        self.user_mean = {u: user_sum[u] / user_cnt[u] for u in user_sum}
        self.item_mean = {b: item_sum[b] / item_cnt[b] for b in item_sum}

        idx = list(range(len(triplets)))
        random.shuffle(idx)
        split = int(len(idx) * (1.0 - self.val_ratio))
        train_idx = idx[:split]
        val_idx = idx[split:]

        train_data = [triplets[i] for i in train_idx]
        val_data = [triplets[i] for i in val_idx]

        base_lr = self.lr
        best_val_rmse = float("inf")
        best_params = None

        print("[Rating] Training MF rating model with early stopping + ensemble...")
        for epoch in range(1, self.n_epochs + 1):
            if epoch in self.lr_decay_steps:
                self.lr *= 0.5
                print(f"[Rating][MF] Decaying learning rate to {self.lr:.5f}")

            random.shuffle(train_data)
            sq_err = 0.0

            for u_i, i_i, r, _, _ in train_data:
                pred = (
                    self.mu
                    + self.b_u[u_i]
                    + self.b_i[i_i]
                    + float(np.dot(self.P[u_i], self.Q[i_i]))
                )
                err = r - pred

                self.b_u[u_i] += self.lr * (err - self.reg * self.b_u[u_i])
                self.b_i[i_i] += self.lr * (err - self.reg * self.b_i[i_i])

                p_u = self.P[u_i]
                q_i = self.Q[i_i]
                self.P[u_i] += self.lr * (err * q_i - self.reg * p_u)
                self.Q[i_i] += self.lr * (err * p_u - self.reg * q_i)

                sq_err += err * err

            train_rmse = math.sqrt(sq_err / len(train_data))

            sq_err_val = 0.0
            for u_i, i_i, r, _, _ in val_data:
                pred = (
                    self.mu
                    + self.b_u[u_i]
                    + self.b_i[i_i]
                    + float(np.dot(self.P[u_i], self.Q[i_i]))
                )
                err = r - pred
                sq_err_val += err * err
            val_rmse = math.sqrt(sq_err_val / len(val_data))

            print(
                f"[Rating][MF] Epoch {epoch}/{self.n_epochs}, "
                f"train RMSE ~ {train_rmse:.4f}, val RMSE ~ {val_rmse:.4f}"
            )

            if val_rmse + 1e-4 < best_val_rmse:
                best_val_rmse = val_rmse
                best_params = (
                    self.P.copy(),
                    self.Q.copy(),
                    self.b_u.copy(),
                    self.b_i.copy(),
                )

        if best_params is not None:
            self.P, self.Q, self.b_u, self.b_i = best_params
            print(
                f"[Rating][MF] Loaded best parameters with val RMSE ~ "
                f"{best_val_rmse:.4f}"
            )

    def _mf_raw(self, u_id, i_id):
        if (u_id not in self.user_index) and (i_id not in self.item_index):
            return self.mu
        elif u_id not in self.user_index:
            i = self.item_index[i_id]
            return self.mu + self.b_i[i]
        elif i_id not in self.item_index:
            u = self.user_index[u_id]
            return self.mu + self.b_u[u]
        else:
            u = self.user_index[u_id]
            i = self.item_index[i_id]
            return (
                self.mu
                + self.b_u[u]
                + self.b_i[i]
                + float(np.dot(self.P[u], self.Q[i]))
            )

    def _baseline_mean(self, u_id, i_id):
        if (u_id in self.user_mean) and (i_id in self.item_mean):
            return self.user_mean[u_id] + self.item_mean[i_id] - self.mu
        elif u_id in self.user_mean:
            return self.user_mean[u_id]
        elif i_id in self.item_mean:
            return self.item_mean[i_id]
        else:
            return self.mu

    def predict(self, u_id, i_id):
        mf_pred = self._mf_raw(u_id, i_id)
        base_pred = self._baseline_mean(u_id, i_id)
        pred = 0.7 * mf_pred + 0.3 * base_pred
        pred = min(5.0, max(1.0, pred))
        return float(pred)


def run_rating_task():
    interactions_path = "train_Interactions.csv.gz"
    pairs_path = "pairs_Rating.csv"
    output_path = "predictions_Rating.csv"

    model = MFWithBias(
        n_factors=40,
        n_epochs=60,
        lr=0.004,
        reg=0.04,
        val_ratio=0.1,
        lr_decay_steps=(30, 45),
    )
    model.fit(interactions_path)

    print("[Rating] Generating predictions...")
    with open(pairs_path, "r", encoding="utf-8") as f_in, \
            open(output_path, "w", encoding="utf-8") as f_out:
        header = f_in.readline().strip()
        f_out.write(header + ",prediction\n")
        for line in f_in:
            u, b = line.strip().split(',')
            pred = model.predict(u, b)
            f_out.write(f"{u},{b},{pred}\n")

    print(f"[Rating] Saved predictions to {output_path}")


# Read prediction

def build_read_indices(interactions_path):
    user_index = {}
    item_index = {}
    user_items = defaultdict(set)
    pos_pairs = []

    for u, b, _ in readCSV(interactions_path):
        if u not in user_index:
            user_index[u] = len(user_index)
        if b not in item_index:
            item_index[b] = len(item_index)
        ui = user_index[u]
        ii = item_index[b]
        user_items[ui].add(ii)
        pos_pairs.append((ui, ii))

    pos_pairs = np.array(pos_pairs, dtype=np.int64)
    return user_index, item_index, user_items, pos_pairs


class ReadTwoTower(nn.Module):
    def __init__(self, n_users, n_items, n_factors=32):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.item_emb = nn.Embedding(n_items, n_factors)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.user_emb.weight, std=0.05)
        nn.init.normal_(self.item_emb.weight, std=0.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_idx, item_idx):
        u_vec = self.user_emb(user_idx)
        i_vec = self.item_emb(item_idx)
        dot = (u_vec * i_vec).sum(dim=1, keepdim=True)
        logits = dot + self.user_bias(user_idx) + self.item_bias(item_idx) + self.global_bias
        return logits.squeeze(1)


def sample_negatives(u_indices, n_items, user_items, neg_per_pos):
    u_neg = np.repeat(u_indices, neg_per_pos)
    i_neg = np.zeros_like(u_neg)
    for j, u in enumerate(u_neg):
        for _ in range(10):
            ii = np.random.randint(0, n_items)
            if ii not in user_items[u]:
                i_neg[j] = ii
                break
        else:
            i_neg[j] = np.random.randint(0, n_items)
    return u_neg, i_neg


def run_read_task():
    interactions_path = "train_Interactions.csv.gz"
    pairs_path = "pairs_Read.csv"
    output_path = "predictions_Read.csv"

    set_global_seeds(0)

    print("[Read] Building user/item indices and positive pairs...")
    user_index, item_index, user_items, pos_pairs = build_read_indices(interactions_path)
    n_users = len(user_index)
    n_items = len(item_index)

    print(f"[Read] #users={n_users}, #items={n_items}, #positive_pairs={len(pos_pairs)}")

    idx_all = np.arange(len(pos_pairs))
    train_idx, val_idx = train_test_split(idx_all, test_size=0.1, random_state=0)
    pos_train = pos_pairs[train_idx]
    pos_val = pos_pairs[val_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReadTwoTower(n_users, n_items, n_factors=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    n_epochs = 10
    batch_size = 4096
    neg_per_pos = 2

    best_val_acc = 0.0
    best_state = None
    best_thr = 0.5

    print("[Read] Training two-tower model with contrastive pairs...")
    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = np.random.permutation(len(pos_train))
        train_correct = 0
        train_total = 0

        for start in range(0, len(pos_train), batch_size):
            batch_idx = perm[start:start + batch_size]
            batch_pos = pos_train[batch_idx]
            u_pos = batch_pos[:, 0]
            i_pos = batch_pos[:, 1]
            bs = len(u_pos)

            u_neg, i_neg = sample_negatives(u_pos, n_items, user_items, neg_per_pos)

            u_all = np.concatenate([u_pos, u_neg])
            i_all = np.concatenate([i_pos, i_neg])
            labels = np.concatenate(
                [np.ones(bs, dtype=np.float32),
                 np.zeros(bs * neg_per_pos, dtype=np.float32)]
            )

            u_all_t = torch.tensor(u_all, dtype=torch.long, device=device)
            i_all_t = torch.tensor(i_all, dtype=torch.long, device=device)
            labels_t = torch.tensor(labels, dtype=torch.float32, device=device)

            optimizer.zero_grad()
            logits = model(u_all_t, i_all_t)
            loss = loss_fn(logits, labels_t)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long()
                train_correct += (preds == labels_t.long()).sum().item()
                train_total += labels_t.size(0)

        train_acc = train_correct / train_total

        model.eval()
        with torch.no_grad():
            u_pos_v = pos_val[:, 0]
            i_pos_v = pos_val[:, 1]
            bs_v = len(u_pos_v)

            u_neg_v, i_neg_v = sample_negatives(u_pos_v, n_items, user_items, 1)

            u_val_all = np.concatenate([u_pos_v, u_neg_v])
            i_val_all = np.concatenate([i_pos_v, i_neg_v])
            labels_val = np.concatenate(
                [np.ones(bs_v, dtype=np.float32),
                 np.zeros(bs_v, dtype=np.float32)]
            )

            u_val_t = torch.tensor(u_val_all, dtype=torch.long, device=device)
            i_val_t = torch.tensor(i_val_all, dtype=torch.long, device=device)
            labels_val_t = torch.tensor(labels_val, dtype=torch.float32, device=device)

            logits_val = model(u_val_t, i_val_t)
            probs_val = torch.sigmoid(logits_val)

            best_epoch_thr = 0.5
            best_epoch_acc = 0.0
            for thr in np.linspace(0.3, 0.7, 17):
                preds_thr = (probs_val >= thr).long()
                acc_thr = (preds_thr == labels_val_t.long()).sum().item() / labels_val_t.size(0)
                if acc_thr > best_epoch_acc:
                    best_epoch_acc = acc_thr
                    best_epoch_thr = thr

        print(
            f"[Read] Epoch {epoch}/{n_epochs}, "
            f"train_acc ~ {train_acc:.4f}, val_acc(best thr) ~ {best_epoch_acc:.4f} "
            f"at thr={best_epoch_thr:.3f}"
        )

        if best_epoch_acc > best_val_acc + 1e-4:
            best_val_acc = best_epoch_acc
            best_thr = float(best_epoch_thr)
            best_state = copy.deepcopy(model.state_dict())

        if epoch in (5, 8):
            for g in optimizer.param_groups:
                g["lr"] *= 0.5
                print(f"[Read] Decayed learning rate to {g['lr']:.5f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(
            f"[Read] Loaded best model with val_acc ~ {best_val_acc:.4f}, "
            f"best threshold={best_thr:.3f}"
        )

    model.eval()

    def predict_read(u_id, b_id):
        if (u_id not in user_index) or (b_id not in item_index):
            return 0
        ui = user_index[u_id]
        ii = item_index[b_id]
        u_t = torch.tensor([ui], dtype=torch.long, device=device)
        i_t = torch.tensor([ii], dtype=torch.long, device=device)
        with torch.no_grad():
            logit = model(u_t, i_t)
            prob = torch.sigmoid(logit).item()
        return 1 if prob >= best_thr else 0

    print("[Read] Predicting on test pairs...")
    with open(pairs_path, "r", encoding="utf-8") as f_in, \
            open(output_path, "w", encoding="utf-8") as f_out:
        header = f_in.readline().strip()
        f_out.write(header + ",prediction\n")
        for line in f_in:
            u, b = line.strip().split(',')
            label = predict_read(u, b)
            f_out.write(f"{u},{b},{label}\n")

    print(f"[Read] Saved predictions to {output_path}")

# Category prediction 

def run_category_task():
    train_path = "train_Category.json.gz"
    test_path = "test_Category.json.gz"
    output_path = "predictions_Category.csv"

    set_global_seeds(0)
    print("[Category] Loading training and test data...")

    train_texts = []
    train_labels = []
    cat_dict = {}
    for obj in readGz(train_path):
        genre = obj["genre"]
        genre_id = int(obj["genreID"])
        train_texts.append(obj["review_text"])
        train_labels.append(genre_id)
        if genre not in cat_dict:
            cat_dict[genre] = genre_id

    num_classes = len(set(train_labels))
    train_labels = np.array(train_labels, dtype=np.int64)
    print(
        f"[Category] #train_samples = {len(train_texts)}, "
        f"#classes = {num_classes}, categories: {cat_dict}"
    )

    test_texts = []
    test_user_ids = []
    test_review_ids = []
    for obj in readGz(test_path):
        test_texts.append(obj["review_text"])
        test_user_ids.append(obj["user_id"])
        test_review_ids.append(obj["review_id"])
    print(f"[Category] #test_samples = {len(test_texts)}")

    train_texts = [t.lower() for t in train_texts]
    test_texts = [t.lower() for t in test_texts]

    from sklearn.feature_extraction.text import TfidfVectorizer

    print("[Category] Building TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=150000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_texts)
    print(f"[Category] TF-IDF matrix shape (train): {X_train.shape}")

    X_test_full = vectorizer.transform(test_texts)

    print("[Category] Train/validation split for model selection...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, train_labels, test_size=0.1, random_state=0, stratify=train_labels
    )

    print("[Category] Training Logistic Regression classifier...")
    clf = LogisticRegression(
        max_iter=200,
        C=4.0,
        multi_class="multinomial",
        solver="lbfgs",
        n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)
    val_pred = clf.predict(X_val)
    val_acc = (val_pred == y_val).mean()
    print(f"[Category] Validation accuracy (TF-IDF + multinomial LR): {val_acc:.4f}")

    print("[Category] Re-training on full training set...")
    clf_full = LogisticRegression(
        max_iter=200,
        C=4.0,
        multi_class="multinomial",
        solver="lbfgs",
        n_jobs=-1,
    )
    clf_full.fit(X_train, train_labels)

    print("[Category] Transforming test data and generating predictions...")
    test_pred = clf_full.predict(X_test_full)

    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write("userID,reviewID,prediction\n")
        for uid, rid, pred in zip(test_user_ids, test_review_ids, test_pred):
            f_out.write(f"{uid},{rid},{int(pred)}\n")
    print(f"[Category] Predictions saved to {output_path}")


# Main entry

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task",
        choices=["rating", "read", "category", "all"],
        help="Which task to run.",
    )
    args = parser.parse_args()
    if args.task in ("rating", "all"):
        run_rating_task()
    if args.task in ("read", "all"):
        run_read_task()
    if args.task in ("category", "all"):
        run_category_task()


if __name__ == "__main__":
    main()
