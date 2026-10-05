# ==============================================================================
# SPP-FLIDS : SUPPLEMENTARY runner  (paste into ONE Kaggle notebook cell)
# Dataset needed in the notebook: Edge-IIoTset Cyber Security Dataset of IoT & IIoT
#
#   QUICK_TEST = True   -> smoke test, about 5 minutes (tiny data, 2 rounds)
#   QUICK_TEST = False  -> full run, about 1 hour (use "Save & Run All (Commit)")
#
# Adds: (1) per-client class distributions, (2) PR-AUC / ROC-AUC and curves,
#       (3) measured client/server timing, (4) reproducibility check vs results.json
# Output: /kaggle/working/supplement.json   (download it and send it back)
# ==============================================================================
QUICK_TEST = True

import warnings; warnings.filterwarnings('ignore')
import glob, time, copy, json, collections
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix, average_precision_score, roc_auc_score, precision_recall_curve, roc_curve

OUT = '/kaggle/working'
if QUICK_TEST:
    ROUNDS, LOCAL_EP, CEN_EP, SEEDS = 2, 1, 2, [0]
else:
    ROUNDS, LOCAL_EP, CEN_EP, SEEDS = 20, 5, 45, [0, 1, 2]
K, LR, DELTA, MU_STAR = 10, 0.05, 1e-5, 0.001
SIGMAS = [0.03, 0.05, 0.1, 0.3]
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print("QUICK_TEST =", QUICK_TEST, "| device =", dev, "| rounds =", ROUNDS, "| local epochs =", LOCAL_EP, "| seeds =", SEEDS)

# ----------------------------- data -------------------------------------------
paths = glob.glob('/kaggle/input/**/*.csv', recursive=True)
path = [p for p in paths if p.endswith('ML-EdgeIIoT-dataset.csv')][0]
DROP_COLS = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
             "http.file_data", "http.request.full_uri", "icmp.transmit_timestamp", "http.request.uri.query",
             "tcp.options", "tcp.payload", "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.msg"]
all_cols = pd.read_csv(path, nrows=5).columns.tolist()
keep = [c for c in all_cols if c not in DROP_COLS]
df = pd.concat([c for c in pd.read_csv(path, usecols=keep, chunksize=50000)], ignore_index=True)
raw_rows = len(df)
df = df.drop_duplicates().dropna().reset_index(drop=True)
clean_rows = len(df)
if QUICK_TEST:
    df = df.sample(n=20000, random_state=0).reset_index(drop=True)
print("rows raw / cleaned / used:", raw_rows, clean_rows, len(df))

X = df.drop(columns=['Attack_label', 'Attack_type'])
for c in X.columns:
    if X[c].dtype == 'object':
        X[c] = X[c].astype(str).astype('category').cat.codes
X = X.astype('float32').replace([np.inf, -np.inf], 0).values
le = LabelEncoder(); ym = le.fit_transform(df['Attack_type']); classes = list(le.classes_); NCLS = len(classes)

Xtr_raw, Xte_raw, ytr, yte = train_test_split(X, ym, test_size=0.2, stratify=ym, random_state=42)
tr_i, va_i = train_test_split(np.arange(len(ytr)), test_size=0.1, stratify=ytr, random_state=42)
scaler = StandardScaler().fit(Xtr_raw[tr_i])                      # statistics from the training part only
T = lambda a: torch.tensor(scaler.transform(a), dtype=torch.float32)
Xs, ys = T(Xtr_raw[tr_i]), torch.tensor(ytr[tr_i], dtype=torch.long)
Xv, yv = T(Xtr_raw[va_i]), torch.tensor(ytr[va_i], dtype=torch.long)
Xte, yte_t = T(Xte_raw), torch.tensor(yte, dtype=torch.long)
d = Xs.shape[1]
print("train / val / test:", len(ys), len(yv), len(yte_t), "| features:", d, "| classes:", NCLS)

# ----------------------------- numpy-only utilities (begin) -------------------
def secure_sum(vecs, round_seed):
    """Simulated pairwise-masking secure aggregation: masks cancel in the sum."""
    Kc = len(vecs); n = vecs[0].size
    masked = [v.astype(np.float64) for v in vecs]
    for i in range(Kc):
        for j in range(i + 1, Kc):
            rs = np.random.RandomState((round_seed * 7919 + i * 104729 + j * 1299709) % (2 ** 32 - 1))
            r = rs.standard_normal(n)
            masked[i] += r; masked[j] -= r
    total = np.sum(masked, axis=0)
    err = float(np.max(np.abs(total - np.sum([v.astype(np.float64) for v in vecs], axis=0))))
    return total, err

def eps_gaussian(sigma, Kc, Tr, delta=1e-5):
    """RDP of the Gaussian mechanism composed over Tr rounds (full participation).
    Effective noise multiplier = sigma*sqrt(Kc) when the Kc client noises are summed (secure aggregation)."""
    if sigma <= 0: return float('inf')
    s2 = sigma ** 2 * Kc
    a = np.arange(1.01, 3000, 0.05)
    return float(np.min(Tr * a / (2 * s2) + np.log(1 / delta) / (a - 1)))

def iid_partition(y, Kc, seed=0):
    rng = np.random.RandomState(seed)
    return list(np.array_split(rng.permutation(len(y)), Kc))

def dirichlet_partition(y, Kc, alpha, seed=0, min_size=50, max_tries=500):
    rng = np.random.RandomState(seed)
    for _ in range(max_tries):
        parts = [[] for _ in range(Kc)]
        for c in np.unique(y):
            idx_c = np.where(y == c)[0]; rng.shuffle(idx_c)
            p = rng.dirichlet(alpha * np.ones(Kc))
            cuts = (np.cumsum(p) * len(idx_c)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(idx_c, cuts)):
                parts[k] += chunk.tolist()
        if min(len(p) for p in parts) >= min_size:
            return [np.array(p) for p in parts]
    return dirichlet_partition(y, Kc, alpha, seed + 1000, max(1, min_size // 2), max_tries)
# ----------------------------- numpy-only utilities (end) ---------------------

# ----------------------------- model / training -------------------------------
loss_fn = nn.CrossEntropyLoss()
def make_model(d, k):
    return nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, k))

def predict(gm, Xin):
    gm.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(Xin), 8192):
            out.append(gm(Xin[i:i + 8192].to(dev)).argmax(1).cpu())
    return torch.cat(out).numpy()

def score(gm, Xin, yin):
    p = predict(gm, Xin); yy = yin.numpy()
    return float(f1_score(yy, p, average='macro')), float((p == yy).mean()), p

def local_train(gm, Xl, yl, epochs, lr, mu, bs=256):
    m = copy.deepcopy(gm); m.train()
    gparams = [p.detach().clone() for p in gm.parameters()]
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9)
    dl = DataLoader(TensorDataset(Xl, yl), batch_size=bs, shuffle=True)
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss_fn(m(xb), yb).backward()
            if mu > 0:
                for p, g in zip(m.parameters(), gparams):
                    p.grad += mu * (p.data - g)
            opt.step()
    return m.state_dict()

def get_vec(sd, names):
    return torch.cat([sd[n].detach().flatten().cpu() for n in names])

def set_from_vec(model, vec, names):
    sd = model.state_dict(); i = 0; new = {}
    for n in names:
        k = sd[n].numel()
        new[n] = vec[i:i + k].view_as(sd[n]).to(sd[n].device); i += k
    model.load_state_dict(new)

def fed_run(parts, seed, mu=0.0, clipC=None, noise_std=0.0, bad=(), secagg=False):
    torch.manual_seed(seed); np.random.seed(seed)
    gen = torch.Generator().manual_seed(1000 + seed)
    gm = make_model(d, NCLS).to(dev)
    names = list(gm.state_dict().keys())
    hist, norm_hist, sec_t, sec_err, diverged = [], [], 0.0, 0.0, False
    for r in range(ROUNDS):
        g_vec = get_vec(gm.state_dict(), names)
        deltas, norms = [], []
        for k, p in enumerate(parts):
            idx = torch.from_numpy(p); yk = ys[idx]
            if k in bad: yk = (yk + 1) % NCLS                       # label-flipping client
            sd = local_train(gm, Xs[idx], yk, LOCAL_EP, LR, mu)
            dl = get_vec(sd, names) - g_vec
            nrm = dl.norm().item(); norms.append(nrm)
            if clipC is not None: dl = dl * min(1.0, clipC / (nrm + 1e-12))
            if noise_std > 0: dl = dl + torch.randn(dl.shape, generator=gen) * noise_std
            deltas.append(dl)
        if secagg:
            t0 = time.time()
            tot, err = secure_sum([x.numpy() for x in deltas], seed * 100003 + r)
            sec_t += time.time() - t0; sec_err = max(sec_err, err)
            avg = torch.from_numpy(tot / len(parts)).float()
        else:
            avg = torch.stack(deltas).mean(0)                          # uniform average
        new_vec = g_vec + avg
        if not torch.isfinite(new_vec).all():                              # training collapsed (NaN/inf weights)
            diverged = True
            hist += [hist[-1] if hist else 0.0] * (ROUNDS - len(hist))
            norm_hist += [None] * (ROUNDS - len(norm_hist))
            break                                                           # keep the last finite global model
        set_from_vec(gm, new_vec, names)
        hist.append(score(gm, Xv, yv)[0])
        mnorm = float(np.median(norms)); norm_hist.append(mnorm if np.isfinite(mnorm) else None)
    return gm, hist, norm_hist, sec_t, sec_err, diverged


# ----------------------------- supplementary measurements ---------------------
EXPECTED = {'central': [0.791117, 0.796369, 0.798417], 'iid': [0.786226, 0.793038, 0.795376],
            'a05': [0.730428, 0.76798, 0.77649], 'a01': [0.651428, 0.713857, 0.569277]}   # test macro-F1 from results.json
SUP = {'meta': dict(quick=QUICK_TEST, rounds=ROUNDS, local_epochs=LOCAL_EP, seeds=SEEDS, K=K, classes=classes), 'partitions': {}, 'runs': [], 'curves': {}}
GRID = np.linspace(0, 1, 101)
def save(): json.dump(SUP, open(OUT + '/supplement.json', 'w'))

def proba(gm, Xin):
    gm.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(Xin), 8192):
            out.append(torch.softmax(gm(Xin[i:i + 8192].to(dev)), 1).cpu())
    return torch.cat(out).numpy()

def curves_and_auc(P, yy):
    Y = np.eye(NCLS)[yy]; ap, auc, pr_c, roc_c = [], [], [], []
    for c in range(NCLS):
        ap.append(float(average_precision_score(Y[:, c], P[:, c]))); auc.append(float(roc_auc_score(Y[:, c], P[:, c])))
        p, r, _ = precision_recall_curve(Y[:, c], P[:, c])                    # r decreasing
        pr_c.append([float(np.max(p[r >= g])) if np.any(r >= g) else 0.0 for g in GRID])
        fpr, tpr, _ = roc_curve(Y[:, c], P[:, c]); roc_c.append([float(v) for v in np.interp(GRID, fpr, tpr)])
    return ap, auc, pr_c, roc_c

# 1) per-client class distributions (seeds 0-2, alpha 0.5 and 0.1)
ynp = ys.numpy()
for a in [0.5, 0.1]:
    SUP['partitions'][str(a)] = {}
    for s in SEEDS:
        parts = dirichlet_partition(ynp, K, a, seed=s)
        SUP['partitions'][str(a)][str(s)] = dict(sizes=[int(len(p)) for p in parts],
            class_counts=[np.bincount(ynp[p], minlength=NCLS).tolist() for p in parts])
save(); print("partition statistics saved", flush=True)

# 2) timing-instrumented federated run (same logic as the main experiments, FedAvg, no clipping/noise)
def fed_run_t(parts, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    gm = make_model(d, NCLS).to(dev); names = list(gm.state_dict().keys())
    hist, client_t, agg_t, round_t = [], [], [], []
    for r in range(ROUNDS):
        t_round = time.perf_counter(); g_vec = get_vec(gm.state_dict(), names); deltas = []; ct = []
        for k, p in enumerate(parts):
            idx = torch.from_numpy(p); t0 = time.perf_counter()
            sd = local_train(gm, Xs[idx], ys[idx], LOCAL_EP, LR, 0.0)
            deltas.append(get_vec(sd, names) - g_vec); ct.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); avg = torch.stack(deltas).mean(0); set_from_vec(gm, g_vec + avg, names); agg_t.append(time.perf_counter() - t0)
        client_t.append(ct); hist.append(score(gm, Xv, yv)[0]); round_t.append(time.perf_counter() - t_round)
    return gm, hist, client_t, agg_t, round_t

def record(name, seed, gm, hist, timing):
    P = proba(gm, Xte); yy = yte_t.numpy(); pred = P.argmax(1)
    ap, auc, pr_c, roc_c = curves_and_auc(P, yy); f1t = float(f1_score(yy, pred, average='macro'))
    exp = EXPECTED[name][seed] if seed < len(EXPECTED[name]) else None
    SUP['runs'].append(dict(config=name, seed=seed, test_f1=f1t, expected_f1=exp, repro_diff=(None if exp is None else f1t - exp),
        macro_pr_auc=float(np.mean(ap)), macro_roc_auc=float(np.mean(auc)), ap_per_class=ap, roc_auc_per_class=auc,
        val_hist=[float(x) for x in hist], timing=timing))
    if seed == SEEDS[0]:
        SUP['curves'][name] = dict(pr=pr_c, roc=roc_c, grid=[float(g) for g in GRID])
    save(); print(f"[{name}] seed={seed} | test F1 {f1t:.4f} (expected {exp}) | macro PR-AUC {np.mean(ap):.4f} | macro ROC-AUC {np.mean(auc):.4f}", flush=True)

t_start = time.time()
for s in SEEDS:
    torch.manual_seed(s); t0 = time.time()
    m = make_model(d, NCLS).to(dev); opt = torch.optim.Adam(m.parameters(), lr=1e-3); sched = torch.optim.lr_scheduler.StepLR(opt, 15, 0.5)
    dl = DataLoader(TensorDataset(Xs, ys), batch_size=256, shuffle=True)
    for ep in range(CEN_EP):
        m.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev); opt.zero_grad(); loss_fn(m(xb), yb).backward(); opt.step()
        sched.step()
    record('central', s, m, [], dict(train_seconds=time.time() - t0))
    for name, alpha in [('iid', 'iid'), ('a05', 0.5), ('a01', 0.1)]:
        parts = iid_partition(ynp, K, s) if alpha == 'iid' else dirichlet_partition(ynp, K, alpha, seed=s)
        gm, hist, client_t, agg_t, round_t = fed_run_t(parts, s)
        flat = [x for r_ in client_t for x in r_]
        n_par = sum(p.numel() for p in gm.parameters())
        timing = dict(client_round_seconds_mean=float(np.mean(flat)), client_round_seconds_median=float(np.median(flat)), client_round_seconds_max=float(np.max(flat)),
                      server_aggregation_seconds_mean=float(np.mean(agg_t)), round_seconds_mean=float(np.mean(round_t)), total_seconds=float(np.sum(round_t)),
                      uplink_bytes_per_client_per_round=n_par * 4, downlink_bytes_per_client_per_round=n_par * 4,
                      uplink_total_bytes_all_clients=n_par * 4 * K * ROUNDS, rounds_to_95pct_of_final=int(next((i + 1 for i, v in enumerate(hist) if v >= 0.95 * hist[-1]), ROUNDS)))
        record(name, s, gm, hist, timing)

# 3) micro-benchmark of the client-side privacy operations (clip + noise) on one update vector
v = torch.randn(46863) * 0.01; gen = torch.Generator().manual_seed(0); t0 = time.perf_counter()
for _ in range(1000):
    u = v * min(1.0, 2.17 / (v.norm().item() + 1e-12)); u = u + torch.randn(u.shape, generator=gen) * (0.05 * 2.17)
SUP['clip_noise_seconds_per_update'] = (time.perf_counter() - t0) / 1000
save()

print("\n=========== SUPPLEMENT SUMMARY ===========")
for name in ['central', 'iid', 'a05', 'a01']:
    rs = [r for r in SUP['runs'] if r['config'] == name]
    print(name, "| macro PR-AUC %.4f | macro ROC-AUC %.4f | max |F1 diff vs results.json| %.5f" % (np.mean([r['macro_pr_auc'] for r in rs]), np.mean([r['macro_roc_auc'] for r in rs]), max(abs(r['repro_diff']) for r in rs if r['repro_diff'] is not None)))
print("total minutes:", round((time.time() - t_start) / 60, 1), "| saved:", OUT + '/supplement.json')
