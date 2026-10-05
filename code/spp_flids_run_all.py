# ==============================================================================
# SPP-FLIDS : one-shot experiment runner  (paste into ONE Kaggle notebook cell)
# Dataset needed in the notebook: Edge-IIoTset Cyber Security Dataset of IoT & IIoT
#
#   QUICK_TEST = True   -> smoke test, about 10 minutes (tiny data, 2 rounds)
#   QUICK_TEST = False  -> full run, about 4-6 hours (use "Save & Run All (Commit)")
#
# Output: /kaggle/working/results.json   (download it and send it back)
# ==============================================================================
QUICK_TEST = True

import warnings; warnings.filterwarnings('ignore')
import glob, time, copy, json, collections
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix

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

# ----------------------------- bookkeeping ------------------------------------
R, META, _cache, _pc = [], {}, {}, {}
C_STAR, SIGMA_STAR = None, None
def save():
    META.update(dict(quick=QUICK_TEST, rounds=ROUNDS, local_epochs=LOCAL_EP, seeds=SEEDS, K=K, lr=LR, delta=DELTA,
                     mu_star=MU_STAR, C_star=C_STAR, sigma_star=SIGMA_STAR, classes=classes, n_features=d,
                     rows_raw=raw_rows, rows_clean=clean_rows, n_train=len(ys), n_val=len(yv), n_test=len(yte_t),
                     test_support=np.bincount(yte, minlength=NCLS).tolist(), dropped_columns=DROP_COLS))
    json.dump({'meta': META, 'results': R}, open(OUT + '/results.json', 'w'))

def get_parts(alpha, seed):
    if (alpha, seed) not in _pc:
        ynp = ys.numpy()
        _pc[(alpha, seed)] = iid_partition(ynp, K, seed) if alpha == 'iid' else dirichlet_partition(ynp, K, alpha, seed)
    return _pc[(alpha, seed)]

def run_cfg(exp, method, alpha, seed, mu=0.0, clip=False, sigma=0.0, mal=0, secagg=False):
    key = (alpha, seed, mu, clip, sigma, mal)              # secure aggregation does not change the aggregate
    if key not in _cache:
        t0 = time.time()
        clipC = C_STAR if clip else None
        noise_std = sigma * C_STAR if sigma > 0 else 0.0
        gm, hist, nh, sec_t, sec_err, div = fed_run(get_parts(alpha, seed), seed, mu, clipC, noise_std, tuple(range(mal)), secagg)
        f1t, acc, pred = score(gm, Xte, yte_t); yy = yte_t.numpy()
        _cache[key] = dict(alpha=str(alpha), seed=seed, mu=mu, clip=clip, sigma=sigma, mal=mal, secagg=secagg,
                           val_f1=float(hist[-1]), val_hist=[float(x) for x in hist], test_f1=f1t, test_acc=acc,
                           per_class_f1=[float(x) for x in f1_score(yy, pred, average=None, labels=list(range(NCLS)))],
                           cm=confusion_matrix(yy, pred, labels=list(range(NCLS))).tolist(),
                           asr=float(np.mean(pred == ((yy + 1) % NCLS))), norm_hist=nh,
                           sec_time=sec_t, sec_err=sec_err, diverged=div, secs=time.time() - t0)
        print(f"[{exp}] {method} | alpha={alpha} seed={seed} | val {hist[-1]:.4f} test {f1t:.4f}{' DIVERGED' if div else ''} | {time.time()-t0:.0f}s", flush=True)
    rec = dict(_cache[key]); rec.update(exp=exp, method=method)
    if sigma > 0 and clip:
        rec['eps_central'] = eps_gaussian(sigma, K, ROUNDS, DELTA)      # noise summed under secure aggregation
        rec['eps_local'] = eps_gaussian(sigma, 1, ROUNDS, DELTA)        # server sees each noisy update (no SecAgg)
    R.append(rec); save()
    return rec

def central_run(seed):
    torch.manual_seed(seed); t0 = time.time()
    m = make_model(d, NCLS).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3); sched = torch.optim.lr_scheduler.StepLR(opt, 15, 0.5)
    dl = DataLoader(TensorDataset(Xs, ys), batch_size=256, shuffle=True)
    for ep in range(CEN_EP):
        m.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); loss_fn(m(xb), yb).backward(); opt.step()
        sched.step()
    f1v = score(m, Xv, yv)[0]; f1t, acc, pred = score(m, Xte, yte_t); yy = yte_t.numpy()
    R.append(dict(exp='central', method='Centralized', alpha='none', seed=seed, mu=0.0, clip=False, sigma=0.0, mal=0,
                  secagg=False, val_f1=f1v, test_f1=f1t, test_acc=acc,
                  per_class_f1=[float(x) for x in f1_score(yy, pred, average=None, labels=list(range(NCLS)))],
                  cm=confusion_matrix(yy, pred, labels=list(range(NCLS))).tolist(), asr=float(np.mean(pred == ((yy + 1) % NCLS)))))
    save(); print(f"[central] seed={seed} | val {f1v:.4f} test {f1t:.4f} | {time.time()-t0:.0f}s", flush=True)

# ----------------------------- experiments ------------------------------------
t_start = time.time()
for s in SEEDS: central_run(s)

# 1) heterogeneity (FedAvg vs FedProx) -- also provides the clipping bound C
r0 = run_cfg('hetero', 'FedAvg', 0.5, SEEDS[0])
C_STAR = round(float(np.median(r0['norm_hist'][:5])), 4)
print("Clipping bound C =", C_STAR)
for s in SEEDS:
    for a in ['iid', 0.5, 0.1]:
        run_cfg('hetero', 'FedAvg', a, s)
        run_cfg('hetero', 'FedProx mu=0.001', a, s, mu=MU_STAR)
    run_cfg('hetero', 'FedProx mu=0.01', 0.1, s, mu=0.01)

# 2) privacy-utility curve (alpha = 0.5)
for s in SEEDS:
    run_cfg('dp', 'Clip only (sigma=0)', 0.5, s, clip=True)
    for sg in SIGMAS:
        run_cfg('dp', f'Clip+DP sigma={sg}', 0.5, s, clip=True, sigma=sg)
mv = lambda sg: float(np.mean([r['val_f1'] for r in R if r['exp'] == 'dp' and r['sigma'] == sg]))
dv = lambda sg: any(r['diverged'] for r in R if r['exp'] == 'dp' and r['sigma'] == sg)
ok = [sg for sg in SIGMAS if mv(sg) >= mv(0.0) - 0.05 and not dv(sg)]   # chosen on VALIDATION only
SIGMA_STAR = max(ok) if ok else min(SIGMAS)
print("Selected sigma* (validation, <= 0.05 F1 loss vs clip-only):", SIGMA_STAR)

# 3) ablation (alpha = 0.1) -- Full uses the simulated secure aggregation
for s in SEEDS:
    run_cfg('ablation', 'Full SPP-FLIDS', 0.1, s, mu=MU_STAR, clip=True, sigma=SIGMA_STAR, secagg=True)
    run_cfg('ablation', '- FedProx', 0.1, s, clip=True, sigma=SIGMA_STAR)
    run_cfg('ablation', '- Clipping', 0.1, s, mu=MU_STAR, clip=False, sigma=SIGMA_STAR)
    run_cfg('ablation', '- DP noise', 0.1, s, mu=MU_STAR, clip=True)
    run_cfg('ablation', '- SecAgg', 0.1, s, mu=MU_STAR, clip=True, sigma=SIGMA_STAR)

# 4) malicious clients (label flipping, alpha = 0.5)
for s in SEEDS:
    for m_ in [0, 1, 2]:
        run_cfg('malicious', 'FedAvg', 0.5, s, mal=m_)
        run_cfg('malicious', 'SPP-FLIDS', 0.5, s, mu=MU_STAR, clip=True, sigma=SIGMA_STAR, mal=m_)

# 5) secure-aggregation overhead benchmark (simulation + analytical communication model)
n_params = sum(p.numel() for p in make_model(d, NCLS).parameters())
vecs = [(np.random.randn(n_params) * 0.01).astype(np.float32) for _ in range(K)]
t0 = time.time(); tot, err = secure_sum(vecs, 0); t_sec = time.time() - t0
t0 = time.time(); plain = np.sum([v.astype(np.float64) for v in vecs], axis=0); t_plain = time.time() - t0
META['secagg_benchmark'] = dict(n_params=n_params, K=K, masked_sum_time_s=t_sec, plain_sum_time_s=t_plain,
    max_abs_error=err, payload_bytes_per_client=n_params * 4,
    modelled_extra_bytes_per_client=64 + 64 * (K - 1) + 2 * 96 * (K - 1),
    assumption="32-byte keys, 2 public keys per client, 2 secret shares of 48 bytes per peer; analytical model, not a deployed protocol")
save()

# ----------------------------- summary ----------------------------------------
g = collections.defaultdict(list)
for r in R: g[(r['exp'], r['method'], r['alpha'], r['sigma'], r['mal'])].append(r['test_f1'])
print("\n=========== SUMMARY (test macro-F1, mean +- std over seeds) ===========")
for k, v in g.items():
    print(k, f"{np.mean(v):.4f} +- {(np.std(v, ddof=1) if len(v) > 1 else 0):.4f} (n={len(v)})")
print("C =", C_STAR, "| sigma* =", SIGMA_STAR, "| total minutes:", round((time.time() - t_start) / 60, 1))
print("Saved:", OUT + '/results.json')
