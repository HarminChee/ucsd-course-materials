# homework2.py — Haomin Qi

from __future__ import annotations
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Iterable, Callable, Any
from collections import defaultdict
from sklearn.linear_model import LogisticRegression

from importlib import reload
import homework2
reload(homework2)


def _ber(tp: int, tn: int, fp: int, fn: int) -> float:
    pos = max(1, tp + fn)
    neg = max(1, tn + fp)
    tpr = tp / pos
    tnr = tn / neg
    return 1.0 - 0.5 * (tpr + tnr)

def _acc(tp: int, tn: int, fp: int, fn: int) -> float:
    tot = tp + tn + fp + fn
    return (tp + tn) / tot if tot > 0 else 0.0

def _bin_counts(y_true: np.ndarray, y_hat: np.ndarray) -> Tuple[int,int,int,int]:
    tp = int(np.sum((y_true == 1) & (y_hat == 1)))
    tn = int(np.sum((y_true == 0) & (y_hat == 0)))
    fp = int(np.sum((y_true == 0) & (y_hat == 1)))
    fn = int(np.sum((y_true == 1) & (y_hat == 0)))
    return tp, tn, fp, fn

def _fit_lr_balanced(X: np.ndarray, y: np.ndarray, C: float) -> LogisticRegression:
    clf = LogisticRegression(class_weight='balanced', C=C, max_iter=1000)
    clf.fit(X, y)
    return clf

def _style_ohe_map(dataset: List[Dict], min_count: int = 1000, style_key: str = "beer/style") -> Dict[str,int]:
    cnt = defaultdict(int)
    for d in dataset:
        s = str(d.get(style_key, ""))
        cnt[s] += 1
    keep = [s for s, c in cnt.items() if c > min_count]
    keep.sort()
    return {s: i for i, s in enumerate(keep)}

def _style_ohe_vec(style: str, style2idx: Dict[str,int]) -> np.ndarray:
    v = np.zeros(len(style2idx), dtype=float)
    if style in style2idx:
        v[style2idx[style]] = 1.0
    return v

def _five_ratings_vec(d: Dict) -> np.ndarray:
    keys = ("review/aroma","review/appearance","review/palate","review/taste","review/overall")
    return np.array([float(d.get(k, 0.0) or 0.0) for k in keys], dtype=float)

def _length_char(d: Dict, text_key: str = "review/text") -> int:
    return len(str(d.get(text_key, "")))

def Q1(a: Any, b: Any = None, c: Any = None, d: Any = None,
       style_key: str = "beer/style", abv_key: str = "beer/ABV",
       C: float = 10.0, min_count: int = 1000):
    if isinstance(a, list):
        train, valid, test = a, b, c
        sty_map = _style_ohe_map(train, min_count=min_count, style_key=style_key)
        def _XY(data: List[Dict]) -> Tuple[np.ndarray,np.ndarray]:
            X = np.vstack([_style_ohe_vec(str(d.get(style_key, "")), sty_map) for d in data]) if len(sty_map) > 0 else np.zeros((len(data), 0))
            y = np.array([1 if float(d.get(abv_key, 0.0)) > 7.0 else 0 for d in data], dtype=int)
            return X, y
        Xtr, ytr = _XY(train); Xva, yva = _XY(valid); Xte, yte = _XY(test)
        clf = _fit_lr_balanced(Xtr, ytr, C=C)
        yva_hat = (clf.predict_proba(Xva)[:, 1] >= 0.5).astype(int)
        tp, tn, fp, fn = _bin_counts(yva, yva_hat)
        acc_val, ber_val = _acc(tp, tn, fp, fn), _ber(tp, tn, fp, fn)
        yte_hat = (clf.predict_proba(Xte)[:, 1] >= 0.5).astype(int)
        tp, tn, fp, fn = _bin_counts(yte, yte_hat)
        acc_test, ber_test = _acc(tp, tn, fp, fn), _ber(tp, tn, fp, fn)
        return acc_val, ber_val, acc_test, ber_test
    else:
        catID, train, valid, test = a, b, c, d
        sty_map = _style_ohe_map(train, min_count=min_count, style_key=style_key)
        def _XY(data: List[Dict]) -> Tuple[np.ndarray,np.ndarray]:
            X = np.vstack([_style_ohe_vec(str(d.get(style_key, "")), sty_map) for d in data]) if len(sty_map) > 0 else np.zeros((len(data), 0))
            y = np.array([1 if float(d.get(abv_key, 0.0)) > 7.0 else 0 for d in data], dtype=int)
            return X, y
        Xtr, ytr = _XY(train); Xva, yva = _XY(valid); Xte, yte = _XY(test)
        clf = _fit_lr_balanced(Xtr, ytr, C=C)
        yva_hat = (clf.predict_proba(Xva)[:, 1] >= 0.5).astype(int)
        tp, tn, fp, fn = _bin_counts(yva, yva_hat); ber_val = _ber(tp, tn, fp, fn)
        yte_hat = (clf.predict_proba(Xte)[:, 1] >= 0.5).astype(int)
        tp, tn, fp, fn = _bin_counts(yte, yte_hat); ber_test = _ber(tp, tn, fp, fn)
        return clf, ber_val, ber_test

def Q2(a, b=None, c=None, d=None,
       style_key: str = "beer/style", abv_key: str = "beer/ABV",
       text_key: str = "review/text",
       C: float = 10.0, min_count: int = 1000):
    if isinstance(a, list):
        train, valid, test = a, b, c
    else:
        train, valid, test = b, c, d
    sty_map = _style_ohe_map(train, min_count=min_count, style_key=style_key)
    maxL = max(1, max(_length_char(d, text_key) for d in train))
    def _fe(datum: Dict) -> np.ndarray:
        return np.concatenate([
            _style_ohe_vec(str(datum.get(style_key, "")), sty_map),
            _five_ratings_vec(datum),
            np.array([_length_char(datum, text_key) / maxL])
        ]).astype(float)
    def _XY(data):
        X = np.vstack([_fe(dd) for dd in data])
        y = np.array([1 if float(dd.get(abv_key, 0.0)) > 7.0 else 0 for dd in data], dtype=int)
        return X, y
    Xtr, ytr = _XY(train); Xva, yva = _XY(valid); Xte, yte = _XY(test)
    clf = _fit_lr_balanced(Xtr, ytr, C=C)
    yva_hat = (clf.predict_proba(Xva)[:, 1] >= 0.5).astype(int)
    tp, tn, fp, fn = _bin_counts(yva, yva_hat); ber_val = _ber(tp, tn, fp, fn)
    yte_hat = (clf.predict_proba(Xte)[:, 1] >= 0.5).astype(int)
    tp, tn, fp, fn = _bin_counts(yte, yte_hat); ber_test = _ber(tp, tn, fp, fn)
    return float(ber_val), float(ber_test)

def Q3(a, b=None, c=None, d=None,
       style_key: str = "beer/style", abv_key: str = "beer/ABV",
       text_key: str = "review/text",
       Cs: Iterable[float] = (0.001, 0.01, 0.1, 1, 10),
       min_count: int = 1000):
    if isinstance(a, list):
        train, valid, test = a, b, c
    else:
        train, valid, test = b, c, d
    sty_map = _style_ohe_map(train, min_count=min_count, style_key=style_key)
    maxL = max(1, max(_length_char(d, text_key) for d in train))
    def _fe(datum: Dict) -> np.ndarray:
        return np.concatenate([
            _style_ohe_vec(str(datum.get(style_key, "")), sty_map),
            _five_ratings_vec(datum),
            np.array([_length_char(datum, text_key) / maxL])
        ]).astype(float)
    def _XY(data):
        X = np.vstack([_fe(dd) for dd in data])
        y = np.array([1 if float(dd.get(abv_key, 0.0)) > 7.0 else 0 for dd in data], dtype=int)
        return X, y
    Xtr, ytr = _XY(train); Xva, yva = _XY(valid); Xte, yte = _XY(test)
    bestC, best_val = None, 1e9
    for C in Cs:
        clf = _fit_lr_balanced(Xtr, ytr, C=C)
        yva_hat = (clf.predict_proba(Xva)[:, 1] >= 0.5).astype(int)
        tp, tn, fp, fn = _bin_counts(yva, yva_hat)
        ber_val = _ber(tp, tn, fp, fn)
        if ber_val < best_val:
            best_val = ber_val; bestC = C
    clf = _fit_lr_balanced(Xtr, ytr, C=bestC)
    yte_hat = (clf.predict_proba(Xte)[:, 1] >= 0.5).astype(int)
    tp, tn, fp, fn = _bin_counts(yte, yte_hat)
    ber_test = _ber(tp, tn, fp, fn)
    return float(bestC), float(best_val), float(ber_test)


def Q4(a, b=None, c=None, d=None,
       style_key: str = "beer/style", abv_key: str = "beer/ABV",
       text_key: str = "review/text",
       C: float = 1.0, min_count: int = 1000):
    if isinstance(a, list):
        train, valid, test = a, b, c
    else:
        train, valid, test = b, c, d
    sty_map = _style_ohe_map(train, min_count=min_count, style_key=style_key)
    maxL = max(1, max(_length_char(d, text_key) for d in train))
    def fe_all(datum: Dict) -> np.ndarray:
        return np.concatenate([
            _style_ohe_vec(str(datum.get(style_key, "")), sty_map),
            _five_ratings_vec(datum),
            np.array([_length_char(datum, text_key) / maxL])
        ]).astype(float)
    n_style = len(sty_map); n_rate = 5
    def _XY_mask(data: List[Dict], drop: str):
        Xf = np.vstack([fe_all(dd) for dd in data])
        if drop == "style":
            X = Xf[:, n_style:]
        elif drop == "ratings":
            X = np.concatenate([Xf[:, :n_style], Xf[:, n_style + n_rate:]], axis=1)
        elif drop == "length":
            X = Xf[:, :n_style + n_rate]
        else:
            X = Xf
        y = np.array([1 if float(dd.get(abv_key, 0.0)) > 7.0 else 0 for dd in data], dtype=int)
        return X, y
    outs = []
    for drop in ("style", "ratings", "length"):
        Xtr, ytr = _XY_mask(train, drop); Xte, yte = _XY_mask(test, drop)
        clf = _fit_lr_balanced(Xtr, ytr, C=C)
        yte_hat = (clf.predict_proba(Xte)[:, 1] >= 0.5).astype(int)
        tp, tn, fp, fn = _bin_counts(yte, yte_hat)
        outs.append(_ber(tp, tn, fp, fn))
    return tuple(float(x) for x in outs)


def _build_index(train: List[Dict], user_key="reviewerID", item_key="asin", rating_key="overall"):
    U_items = defaultdict(list)
    I_users = defaultdict(list)
    item_sum = defaultdict(float); item_cnt = defaultdict(int)
    user_sum = defaultdict(float); user_cnt = defaultdict(int)
    tot_sum = 0.0; tot_cnt = 0
    seen = set()
    for d in train:
        try:
            u = str(d[user_key]); i = str(d[item_key]); r = float(d[rating_key])
        except Exception:
            continue
        if (u, i) in seen:
            continue
        seen.add((u, i))
        U_items[u].append((i, r)); I_users[i].append((u, r))
        item_sum[i] += r; item_cnt[i] += 1
        user_sum[u] += r; user_cnt[u] += 1
        tot_sum += r; tot_cnt += 1
    item_mean = {i: item_sum[i] / item_cnt[i] for i in item_sum}
    user_mean = {u: user_sum[u] / user_cnt[u] for u in user_sum}
    global_mean = tot_sum / tot_cnt if tot_cnt > 0 else 3.5
    return U_items, I_users, item_mean, user_mean, global_mean

def _similarity_adjusted_cosine(i: str, j: str,
                                I_users: Dict[str,List[Tuple[str,float]]],
                                user_mean: Dict[str,float]) -> float:
    users_i = I_users.get(i, [])
    users_j = I_users.get(j, [])
    if not users_i or not users_j:
        return 0.0
    rj = {u: r for (u, r) in users_j}
    xs, ys = [], []
    for u, ri in users_i:
        if u in rj:
            mu = user_mean.get(u, 0.0)
            xs.append(ri - mu); ys.append(rj[u] - mu)
    if not xs:
        return 0.0
    x = np.array(xs); y = np.array(ys)
    nx = np.linalg.norm(x); ny = np.linalg.norm(y)
    if nx == 0 or ny == 0:
        return 0.0
    return float(np.dot(x, y) / (nx * ny))

def Q5(a, b=None, c=None, user_key="reviewerID", item_key="asin", rating_key="overall", topk: int = 10):
    if isinstance(a, list):
        train, query_item = a, b
        U_items, I_users, item_mean, user_mean, gmean = _build_index(train, user_key, item_key, rating_key)
        sims = []
        for j in I_users.keys():
            if j == query_item:
                continue
            s = _similarity_adjusted_cosine(query_item, j, I_users, user_mean)
            if s != 0.0:
                sims.append((s, j))
        sims.sort(key=lambda x: -x[0])
        return sims[:int(topk)]
    else:
        query_item, K, usersPerItem = a, b, c
        res = mostSimilar(query_item, int(K), usersPerItem)
        return [j for _, j in res]


def _predict_item_item(u: str, i: str,
                       U_items: Dict[str,List[Tuple[str,float]]],
                       I_users: Dict[str,List[Tuple[str,float]]],
                       item_mean: Dict[str,float], user_mean: Dict[str,float], gmean: float,
                       kmax: int = 50) -> float:
    if u not in U_items:
        return item_mean.get(i, gmean)
    neigh = []
    for j, r in U_items[u]:
        s = _similarity_adjusted_cosine(i, j, I_users, user_mean)
        if s != 0.0:
            neigh.append((s, r))
    if not neigh:
        return item_mean.get(i, gmean)
    neigh.sort(key=lambda x: -x[0])
    neigh = neigh[:kmax]
    num = sum(s * r for s, r in neigh)
    den = sum(abs(s) for s, _ in neigh)
    if den == 0.0:
        return item_mean.get(i, gmean)
    return num / den

def Q6(train: List[Dict], test: List[Dict],
       user_key="reviewerID", item_key="asin", rating_key="overall") -> float:
    U_items, I_users, item_mean, user_mean, gmean = _build_index(train, user_key, item_key, rating_key)
    sq = 0.0; n = 0
    for d in test:
        try:
            u = str(d[user_key]); i = str(d[item_key]); r = float(d[rating_key])
        except Exception:
            continue
        p = _predict_item_item(u, i, U_items, I_users, item_mean, user_mean, gmean)
        sq += (p - r) ** 2; n += 1
    return math.sqrt(sq / n) if n > 0 else 0.0

def Q7(train: List[Dict], valid: List[Dict], test: List[Dict],
       user_key="reviewerID", item_key="asin", rating_key="overall") -> Tuple[float,float,float]:
    U_items, I_users, item_mean, user_mean, gmean = _build_index(train, user_key, item_key, rating_key)
    def _pred(u,i,alpha):
        base = item_mean.get(i, gmean)
        cf = _predict_item_item(u, i, U_items, I_users, item_mean, user_mean, gmean)
        return alpha * base + (1 - alpha) * cf
    def _rmse(data, alpha):
        sq = 0.0; n = 0
        for d in data:
            try:
                u = str(d[user_key]); i = str(d[item_key]); r = float(d[rating_key])
            except Exception:
                continue
            p = _pred(u, i, alpha); sq += (p - r) ** 2; n += 1
        return math.sqrt(sq / n) if n > 0 else 0.0
    best_a, best_val = None, 1e9
    for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
        rm = _rmse(valid, a)
        if rm < best_val:
            best_val = rm; best_a = a
    rm_te = _rmse(test, best_a)
    return float(best_a), float(best_val), float(rm_te)

def getMeanRating(data: List[Dict], rating_key: str = "overall") -> float:
    s = 0.0; n = 0
    for d in data:
        if rating_key in d:
            try:
                s += float(d[rating_key]); n += 1
            except Exception:
                pass
    return s / n if n > 0 else 0.0

def getItemAverageRatings(data: List[Dict], item_key: str = "asin", rating_key: str = "overall") -> Dict[str, float]:
    s = defaultdict(float); c = defaultdict(int)
    for d in data:
        try:
            i = str(d[item_key]); r = float(d[rating_key])
            s[i] += r; c[i] += 1
        except Exception:
            pass
    return {i: s[i] / c[i] for i in s}

def getUserAverageRatings(data: List[Dict], user_key: str = "reviewerID", rating_key: str = "overall") -> Dict[str, float]:
    s = defaultdict(float); c = defaultdict(int)
    for d in data:
        try:
            u = str(d[user_key]); r = float(d[rating_key])
            s[u] += r; c[u] += 1
        except Exception:
            pass
    return {u: s[u] / c[u] for u in s}

def adjustedCosineSimilarity(a: Any, b: Any = None, c: Any = None,
                             user_key: str = "reviewerID", item_key: str = "asin", rating_key: str = "overall") -> float:
    if isinstance(a, str) and isinstance(b, str) and isinstance(c, dict):
        qi, oj, usersPerItem = a, b, c
        user_sum = defaultdict(float); user_cnt = defaultdict(int)
        for i, users in usersPerItem.items():
            for u in users:
                if isinstance(u, (tuple, list)) and len(u) >= 2:
                    user_sum[u[0]] += float(u[1]); user_cnt[u[0]] += 1
        user_mean = {u: user_sum[u] / user_cnt[u] for u in user_sum if user_cnt[u] > 0}
        Ui = {}
        for u in usersPerItem.get(qi, []):
            if isinstance(u, (tuple, list)) and len(u) >= 2:
                Ui[u[0]] = float(u[1])
        Uj = {}
        for u in usersPerItem.get(oj, []):
            if isinstance(u, (tuple, list)) and len(u) >= 2:
                Uj[u[0]] = float(u[1])
        common = set(Ui.keys()) & set(Uj.keys())
        if not common:
            return 0.0
        xs, ys = [], []
        for u in common:
            mu = user_mean.get(u, None)
            if mu is None:
                continue
            xs.append(Ui[u] - mu); ys.append(Uj[u] - mu)
        if not xs:
            return 0.0
        x = np.array(xs); y = np.array(ys)
        nx = np.linalg.norm(x); ny = np.linalg.norm(y)
        if nx == 0.0 or ny == 0.0:
            return 0.0
        return float(np.dot(x, y) / (nx * ny))
    else:
        data = a; qi = b; oj = c
        U_items, I_users, item_mean, user_mean, gmean = _build_index(data, user_key, item_key, rating_key)
        return _similarity_adjusted_cosine(qi, oj, I_users, user_mean)

def getUserAverages(itemsPerUser: Dict[Any, List[Any]], ratingDict: Dict[Tuple[Any,Any], float]) -> Dict[Any, float]:
    avg = {}
    for u, items in itemsPerUser.items():
        s = 0.0; c = 0
        for it in items:
            if isinstance(it, (tuple, list)) and len(it) >= 2:
                try:
                    r = float(it[1])
                except Exception:
                    r = float(ratingDict.get((u, it[0]), float("nan")))
            else:
                r = float(ratingDict.get((u, it), float("nan")))
            if not math.isnan(r):
                s += r; c += 1
        if c > 0:
            avg[u] = s / c
    return avg

def getItemAverages(usersPerItem: Dict[Any, List[Any]], ratingDict: Dict[Tuple[Any,Any], float]) -> Dict[Any, float]:
    avg = {}
    for i, users in usersPerItem.items():
        s = 0.0; c = 0
        for u in users:
            if isinstance(u, (tuple, list)) and len(u) >= 2:
                try:
                    r = float(u[1])
                except Exception:
                    r = float(ratingDict.get((u[0], i), float("nan")))
            else:
                r = float(ratingDict.get((u, i), float("nan")))
            if not math.isnan(r):
                s += r; c += 1
        if c > 0:
            avg[i] = s / c
    return avg

def mostSimilar(query_item: str, K: int, usersPerItem: Dict[Any, List[Any]]) -> List[Tuple[float,str]]:
    user_sum = defaultdict(float); user_cnt = defaultdict(int)
    for i, users in usersPerItem.items():
        for u in users:
            if isinstance(u, (tuple, list)) and len(u) >= 2:
                user_sum[u[0]] += float(u[1]); user_cnt[u[0]] += 1
    user_mean = {u: user_sum[u] / user_cnt[u] for u in user_sum if user_cnt[u] > 0}
    Ui = {}
    for u in usersPerItem.get(query_item, []):
        if isinstance(u, (tuple, list)) and len(u) >= 2:
            Ui[u[0]] = float(u[1])
    sims = []
    for j, Uj_list in usersPerItem.items():
        if j == query_item:
            continue
        Uj = {}
        for u in Uj_list:
            if isinstance(u, (tuple, list)) and len(u) >= 2:
                Uj[u[0]] = float(u[1])
        common = set(Ui.keys()) & set(Uj.keys())
        if not common:
            continue
        xs, ys = [], []
        for u in common:
            mu = user_mean.get(u, None)
            if mu is None:
                continue
            xs.append(Ui[u] - mu); ys.append(Uj[u] - mu)
        if not xs:
            continue
        x = np.array(xs); y = np.array(ys)
        nx = np.linalg.norm(x); ny = np.linalg.norm(y)
        if nx == 0.0 or ny == 0.0:
            continue
        s = float(np.dot(x, y) / (nx * ny))
        if s != 0.0:
            sims.append((s, j))
    sims.sort(key=lambda t: -t[0])
    return sims[:int(K)]


def predictRating(a, b, c, d=None, e=None, f=None, g=None, h=3.5, k=50):
    if isinstance(c, dict):
        user, item = a, b
        itemsPerUser, usersPerItem, ratingDict = c, d, e
        userAverages = f if isinstance(f, dict) else {}
        itemAverages = g if isinstance(g, dict) else {}
        ratingMean = h if isinstance(h, (int, float)) else 3.5

        if user in itemsPerUser:
            neigh = []
            for j, r in itemsPerUser[user]:
                Uj = usersPerItem.get(j, [])
                Ui = usersPerItem.get(item, [])
                if not Uj or not Ui:
                    continue
                rj = {uu: rr for (uu, rr) in Uj}
                xs, ys = [], []
                for uu, ri in Ui:
                    if uu in rj:
                        mu = userAverages.get(uu, None)
                        if mu is None:
                            continue
                        xs.append(ri - mu); ys.append(rj[uu] - mu)
                if not xs:
                    continue
                x = np.array(xs); y = np.array(ys)
                nx = np.linalg.norm(x); ny = np.linalg.norm(y)
                if nx == 0.0 or ny == 0.0:
                    continue
                s = float(np.dot(x, y) / (nx * ny))
                if s != 0.0:
                    neigh.append((s, r))
            if neigh:
                neigh.sort(key=lambda t: -t[0]); neigh = neigh[:int(k)]
                num = sum(s * r for s, r in neigh)
                den = sum(abs(s) for s, _ in neigh)
                if den != 0.0:
                    return num / den

        if item in itemAverages:
            return float(itemAverages[item])
        if user in userAverages:
            return float(userAverages[user])
        return float(ratingMean)
    else:
        user, item = a, b
        train = c
        U_items, I_users, item_mean, user_mean, gmean = _build_index(train)
        return _predict_item_item(str(user), str(item), U_items, I_users, item_mean, user_mean, gmean, kmax=int(d) if isinstance(d,int) else 50)
