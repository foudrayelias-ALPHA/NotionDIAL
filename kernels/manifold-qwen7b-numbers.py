"""Numbers tab, second 7B: number-line 2-hop answer-manifold dosing on
Qwen2.5-7B (Kaggle T4/P100). Derived from manifold-qwen7b-months.py (the months
version); the NUMBER concept is the edit target, and BOTH the day 2-hop task and
the month/day geometry ride along as the spared readouts on number-edited
weights.

Governed by preregistration_numberstab.md (frozen locally before this kernel
ran). Deviations, all documented in decisions.md: fp16 forward (fp64/fp32 numpy
edit math unchanged); in-place rank-6 edits with exact re-pinning per lambda (no
weight clones); per-lambda number-answer clouds saved for the demo's geometry
panel. Qwen specifics preserved from the day/months versions: single-device load
(device_map={"":0}, assert no meta params); per-device rank-6 edit math computed
in 4096-column fp32 slabs (no full-width clones); wiki log-softmax chunked row by
row over the ~152k vocab. tie status resolved at runtime.

Number specifics vs the months version:
- Concept = 12 number words " one".." twelve" (assert single leading-space token).
- B6n = rank-6 PCA of the 12 number OUTPUT-embedding rows (tied/untied resolved as
  the months kernel does). Frozen before the sweep.
- THE NUMBER LINE DOES NOT WRAP. 2-hop task uses X in one..ten (10 items) so the
  answer X+2 stays in one..twelve; restricted argmax over all 12 number tokens,
  want index i+2, no modular wrap. Successor uses X in one..eleven (11 items),
  want i+1.
- Cloud _A_cloud_B6 = 12 [x,y] points per lambda (number-answer cloud), 4dp.
- Collateral: wiki_kl AND wiki_kl_clean (clean = contexts with no number token;
  number words are frequent in wikitext, report the clean fraction).
- IN-KERNEL ELIGIBILITY GATE: number 2-hop baseline (X in one..ten) >= 8/10 at
  lambda=1; else write ineligible.json and exit.
- SPARED columns: DAY 2-hop task (7 items) behavioral column + day_plane_power +
  month_plane_power geometry readouts, all riding number-edited weights.
- Conditions: T_out_all + C_out_rand_r0 (matched-norm random rank-6, same set).
  r1 also run (runtime permitting).
"""

import json
import subprocess
import sys

gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout
print("GPU:", gpu.strip(), flush=True)
if "P100" in gpu:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                    "torch==2.4.1", "torchvision==0.19.1", "torchaudio==2.4.1",
                    "--index-url", "https://download.pytorch.org/whl/cu121"])

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-7B"
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
NUMBERS = ["one", "two", "three", "four", "five", "six",
           "seven", "eight", "nine", "ten", "eleven", "twelve"]
TEMPLATES = ["The meeting is scheduled for {}.", "She will arrive on {}.",
             "Everything closed last {} evening.", "I always go swimming on {}.",
             "The deadline is next {}.", "It happened one {} morning.",
             "We usually rest on {}.", "The store reopens on {}."]
NUM_TEMPLATES = ["I counted to {}.", "She gave me {} apples.",
                 "There were {} people there.", "He is {} years old.",
                 "We waited for {} hours.", "The score was {} to nothing.",
                 "It takes {} steps.", "They ordered {} coffees."]

FEWSHOT_N = ("Let's do some number math. Two after one is three. Two after five "
             "is seven. Two after {} is")
NSUCC = "One after {} is"
FEWSHOT_M = ("Let's do some month math. Two months after January is "
             "March. Two months after October is December. Two months after {} is")
FEWSHOT_D = ("Let's do some day of the week math. Two days after Monday is "
             "Wednesday. Two days after Friday is Sunday. Two days after {} is")
DSUCC = "If today is {}, then tomorrow is"

NSHIFT, MSHIFT, DSHIFT = 2, 2, 2
LAMBDAS = sorted({-0.5, -0.25} | {round(x, 3) for x in np.linspace(0, 1, 21)}
                 | {1.1, 1.25, 1.5})
OUTDIR = "/kaggle/working"

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map={"": 0}).eval()
assert not any(p.is_meta for p in model.parameters()), "meta tensors present"
print("device map:", getattr(model, "hf_device_map", "single-device {'': 0}"), flush=True)
model.requires_grad_(False)
DEV = "cuda:0"
cfg = model.config
NL, HID = cfg.num_hidden_layers, cfg.hidden_size
PROBE = round(2 * NL / 3)
blocks = model.model.layers
norm = model.model.norm


def tid(words):
    out = []
    for w in words:
        t = tok.encode(" " + w, add_special_tokens=False)
        assert len(t) == 1, (w, t)
        out.append(t[0])
    return out


p_ids = tid(DAYS)
mo_ids = tid(MONTHS)
x_ids = tid(NUMBERS)
assert len(x_ids) == 12 and len(set(x_ids)) == 12, ("number tokens", x_ids)
print("number single-tokens OK:", len(x_ids), flush=True)

oe = model.get_output_embeddings()
ie = model.get_input_embeddings()
tied = oe is None or oe.weight is ie.weight
print("tied embeddings:", tied, flush=True)

W_out_rows = (ie if tied else oe).weight[x_ids].detach().float().cpu().numpy().astype(np.float64)
Xc = W_out_rows - W_out_rows.mean(0)
_, S, Vt = np.linalg.svd(Xc, full_matrices=False)
B6 = np.ascontiguousarray(Vt[:6].T)                     # (HID, 6), frozen number basis
_B6_cache = {}


def B6_on(dev):
    if dev not in _B6_cache:
        _B6_cache[dev] = torch.from_numpy(B6).float().to(dev)
    return _B6_cache[dev]


def positions(tpl_list, words, ids):
    out = []
    for t in tpl_list:
        for wi, w in enumerate(words):
            enc = tok(t.format(w), return_tensors="pt").input_ids
            out.append((enc.to(DEV), wi, enc[0].tolist().index(ids[wi])))
    return out


def ctx_positions(tpl, words):
    return [(tok(tpl.format(w), return_tensors="pt").input_ids.to(DEV), wi, -1)
            for wi, w in enumerate(words)]


day_tpl = positions(TEMPLATES, DAYS, p_ids)
month_tpl = positions(TEMPLATES, MONTHS, mo_ids)
number_tpl = positions(NUM_TEMPLATES, NUMBERS, x_ids)

ntask_enc = ctx_positions(FEWSHOT_N, NUMBERS[:10])    # number 2-hop, 10 items (PRIMARY)
nsucc_enc = ctx_positions(NSUCC, NUMBERS[:11])        # number successor, 11 items
dtask_enc = ctx_positions(FEWSHOT_D, DAYS)            # day 2-hop, 7 items (SPARED behavior)
mtask_enc = ctx_positions(FEWSHOT_M, MONTHS[:10])     # month 2-hop, spared readout

grabbed = {}


def _hook(name):
    def fn(_m, _i, out):
        grabbed[name] = out[0] if isinstance(out, tuple) else out
    return fn


@torch.no_grad()
def cloud(tpl, n_words, layer):
    H = np.zeros((n_words, HID))
    handle = None
    if layer == NL:
        handle = blocks[NL - 1].register_forward_hook(_hook("top"))
    try:
        for enc, wi, pos in tpl:
            hs = model(input_ids=enc, output_hidden_states=True).hidden_states
            src = grabbed["top"] if handle else hs[layer]
            H[wi] += src[0, pos].float().cpu().numpy()
    finally:
        if handle:
            handle.remove()
    return H / (len(tpl) // n_words)


@torch.no_grad()
def restricted_linear(encs, ids, k, shift):
    """Restricted-argmax accuracy over the k concept tokens, target index i+shift,
    NO modular wrap (number line does not wrap); scored over the items whose answer
    index i+shift < k."""
    margins, correct, n = [], 0, 0
    ids_t = torch.tensor(ids, device=DEV)
    for i, (enc, wi, pos) in enumerate(encs):
        want = i + shift
        if want >= k:
            continue
        dl = model(input_ids=enc).logits[0, -1][ids_t].float()
        margins.append(float(dl[want] - dl[torch.arange(k) != want].max()))
        correct += int(dl.argmax()) == want
        n += 1
    return correct / n, float(np.mean(margins)), n


@torch.no_grad()
def restricted_mod(encs, ids, k, shift):
    """Restricted-argmax accuracy with modular wrap (spared cyclic day/month)."""
    margins, correct = [], 0
    ids_t = torch.tensor(ids, device=DEV)
    for i, (enc, wi, pos) in enumerate(encs):
        dl = model(input_ids=enc).logits[0, -1][ids_t].float()
        want = (i + shift) % k
        margins.append(float(dl[want] - dl[torch.arange(k) != want].max()))
        correct += int(dl.argmax()) == want
    return correct / len(encs), float(np.mean(margins))


def pca2(H):
    Hc = H - H.mean(0)
    _, s, vt = np.linalg.svd(Hc, full_matrices=False)
    return vt[:2].T, float((s[:2] ** 2).sum() / (s**2).sum())


def plane_power(H, B):
    Hc = H - H.mean(0)
    return float(((Hc @ B) ** 2).sum())


# ================= ELIGIBILITY GATE (lambda=1, before any edit) =================
n2_acc, n2_margin, n2_n = restricted_linear(ntask_enc, x_ids, 12, NSHIFT)
print(f"GATE number 2-hop baseline: {n2_acc * n2_n:.0f}/{n2_n} (margin {n2_margin:.3f})",
      flush=True)
if n2_acc * n2_n < 8:
    d2_acc, _ = restricted_mod(dtask_enc, p_ids, 7, DSHIFT)
    json.dump({"eligible": False, "number_2hop_baseline_n": int(round(n2_acc * n2_n)),
               "number_2hop_items": n2_n, "number_2hop_baseline": n2_acc,
               "number_2hop_margin": n2_margin, "bar": "8/10",
               "day_2hop_baseline_n": int(round(d2_acc * 7)), "model": MODEL},
              open(f"{OUTDIR}/ineligible.json", "w"))
    print("INELIGIBLE: number 2-hop baseline below 8/10 -> writing ineligible.json, exiting.",
          flush=True)
    sys.exit(0)
print("GATE PASSED (>= 8/10). Proceeding to survey + attribution + sweep.", flush=True)

# ================= STAGE 1: survey =================
from sklearn.linear_model import Ridge

Hn_l = np.zeros((NL + 1, 12, HID))
A_l = np.zeros((NL + 1, n2_n, HID))
handle = blocks[NL - 1].register_forward_hook(_hook("top"))
with torch.no_grad():
    for enc, wi, pos in ntask_enc:
        if wi + NSHIFT >= 12:
            continue
        hs = model(input_ids=enc, output_hidden_states=True).hidden_states
        for l in range(NL):
            A_l[l, wi] = hs[l][0, -1].float().cpu().numpy()
        A_l[NL, wi] = grabbed["top"][0, -1].float().cpu().numpy()
    for enc, wi, pos in number_tpl:
        hs = model(input_ids=enc, output_hidden_states=True).hidden_states
        for l in range(NL):
            Hn_l[l, wi] += hs[l][0, pos].float().cpu().numpy()
        Hn_l[NL, wi] += grabbed["top"][0, pos].float().cpu().numpy()
handle.remove()
Hn_l /= len(NUM_TEMPLATES)
want = [i + NSHIFT for i in range(n2_n)]
Wfull = (ie if tied else oe).weight[x_ids].detach().float().cpu()
lin = np.arange(12, dtype=np.float64)
survey = []
with torch.no_grad():
    for l in range(NL + 1):
        nd = next(norm.parameters()).device
        h = norm(torch.from_numpy(A_l[l]).half().to(nd)).float().cpu()
        dl = h @ Wfull.T
        lens = int(sum(int(dl[i].argmax()) == want[i] for i in range(n2_n)))
        probe = Ridge(alpha=1e-3).fit(Hn_l[l], lin)
        pred = probe.predict(A_l[l])
        dec = [int(round(p)) for p in pred]
        pacc = int(sum(d == w for d, w in zip(dec, want)))
        survey.append({"layer": l, "lens": lens, "probe": pacc})
json.dump({"tied": tied, "rows": survey}, open(f"{OUTDIR}/survey.json", "w"))
print("SURVEY lens", json.dumps([r["lens"] for r in survey]),
      "probe", json.dumps([r["probe"] for r in survey]), flush=True)

# ================= STAGE 2: attribution =================
names = [f"{k}_{l}" for l in range(NL) for k in ("attn", "mlp")]
caps = {n: np.zeros((n2_n, HID)) for n in names}   # 10 non-wrapping 2-hop items
caps["emb"] = np.zeros((n2_n, HID))
total = np.zeros((n2_n, HID))
handles = [blocks[l].self_attn.register_forward_hook(_hook(f"attn_{l}"))
           for l in range(NL)]
handles += [blocks[l].mlp.register_forward_hook(_hook(f"mlp_{l}"))
            for l in range(NL)]
handles.append(blocks[NL - 1].register_forward_hook(_hook("top")))
with torch.no_grad():
    for enc, wi, pos in ntask_enc:
        hs = model(input_ids=enc, output_hidden_states=True).hidden_states
        caps["emb"][wi] += hs[0][0, -1].float().cpu().numpy()
        total[wi] += grabbed["top"][0, -1].float().cpu().numpy()
        for n in names:
            caps[n][wi] += grabbed[n][0, -1].float().cpu().numpy()
for h in handles:
    h.remove()
ssum = sum(caps.values())
rel = float(np.abs(ssum - total).max() / np.abs(total).max())
print("additivity rel err:", rel, flush=True)
Ac = total - total.mean(0)
P6 = B6 @ B6.T
den = float(((Ac @ B6) ** 2).sum())
alphas = {n: float(np.sum(((C - C.mean(0)) @ P6) * (Ac @ P6)) / den)
          for n, C in caps.items()}
late = sum(abs(v) for k, v in alphas.items()
           if k != "emb" and int(k.split("_")[1]) >= 3 * NL // 4)
early = sum(abs(v) for k, v in alphas.items()
            if k != "emb" and int(k.split("_")[1]) < 3 * NL // 4)
top = max((k for k in alphas if k != "emb"), key=lambda k: abs(alphas[k]))
att = {"alphas": dict(sorted(alphas.items(), key=lambda kv: -abs(kv[1]))),
       "late_share": late, "early_share": early, "top_writer": top,
       "additivity_rel_err": rel}
json.dump(att, open(f"{OUTDIR}/attribution.json", "w"))
print("ATTRIB top8", json.dumps(dict(list(att["alphas"].items())[:8])),
      "late", round(late, 3), "early", round(early, 3), flush=True)

# ================= STAGE 3: dosing sweep =================
def proj_weight(l, kind):
    b = blocks[l]
    return (b.self_attn.o_proj if kind == "attn" else b.mlp.down_proj).weight


# per-module original rank-6 content R = B6^T W (fp32, 4096-col slabs, no full clone)
R_orig = {}
for l in range(NL):
    for kind in ("attn", "mlp"):
        W = proj_weight(l, kind)
        Bd = B6_on(W.device)
        parts = [Bd.T @ W[:, c0:min(c0 + 4096, W.shape[1])].float()
                 for c0 in range(0, W.shape[1], 4096)]
        R_orig[f"{kind}_{l}"] = torch.cat(parts, dim=1)   # (6, in), on W's device


def set_lambda(names_set, lam, Q_t32=None, R_rand=None, scale=None):
    """Exact re-pin in 4096-col fp32 slabs: subspace content set to closed form."""
    with torch.no_grad():
        for n in names_set:
            l, kind = int(n.split("_")[1]), n.split("_")[0]
            W = proj_weight(l, kind)
            B6d = B6_on(W.device)
            if Q_t32 is None:  # targeted number plane
                Bd, target = B6d, lam * R_orig[n]
            else:              # norm-matched random control
                Bd = Q_t32[W.device] if isinstance(Q_t32, dict) else Q_t32
                target = (1.0 - (1.0 - lam) * scale[n]) * R_rand[n]
            CH = 4096
            for c0 in range(0, W.shape[1], CH):
                c1 = min(c0 + CH, W.shape[1])
                cur = Bd.T @ W[:, c0:c1].float()
                W[:, c0:c1] += (Bd @ (target[:, c0:c1] - cur)).half()


# ---- wikitext, with clean mask (contexts with NO number token yet) ----
wiki_ids = None
wiki_clean = None
try:
    from datasets import load_dataset
    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n".join(t for t in wiki["text"] if t.strip())[:20000]
    wiki_ids = tok(text, return_tensors="pt").input_ids[0][: 8 * 64].reshape(8, 64).to(DEV)
    tainted = torch.cummax(torch.isin(wiki_ids, torch.tensor(x_ids, device=DEV)), 1).values
    wiki_clean = (~tainted.bool()).cpu()
    print("wiki clean-position fraction:", float(wiki_clean.float().mean()), flush=True)
except Exception as e:
    print("wiki unavailable:", e, flush=True)

Hd = cloud(day_tpl, 7, PROBE)
BHd, _ = pca2(Hd)
day_ref = plane_power(Hd, BHd)
Hmo = cloud(month_tpl, 12, PROBE)
BHmo, _ = pca2(Hmo)
month_ref = plane_power(Hmo, BHmo)
Hn = cloud(number_tpl, 12, PROBE)
BHn, _ = pca2(Hn)
number_ref = plane_power(Hn, BHn)
A_ref_cloud = cloud(ntask_enc, 10, NL)     # number answer cloud (10 non-wrapping pts)
A_ref = plane_power(A_ref_cloud, B6)


def wiki_logp():
    """Chunked (row-by-row) log-softmax over the ~152k vocab to bound memory."""
    outs = []
    with torch.no_grad():
        for r in range(wiki_ids.shape[0]):
            lg = model(input_ids=wiki_ids[r:r + 1]).logits.float()
            outs.append(F.log_softmax(lg, -1).cpu())
    return torch.cat(outs, 0)


if wiki_ids is not None:
    ref_logp = wiki_logp()


def measure():
    m = {}
    Acl = cloud(ntask_enc, 10, NL)         # number answer cloud, 10 non-wrapping points
    m["A_B6_power"] = plane_power(Acl, B6) / max(A_ref, 1e-12)
    m["_A_cloud_B6"] = [[round(float(v), 4) for v in row] for row in
                        ((Acl - Acl.mean(0)) @ B6)[:, :2]]
    m["number_plane_power"] = plane_power(cloud(number_tpl, 12, PROBE), BHn) / max(number_ref, 1e-12)
    m["month_plane_power"] = plane_power(cloud(month_tpl, 12, PROBE), BHmo) / max(month_ref, 1e-12)
    m["day_plane_power"] = plane_power(cloud(day_tpl, 7, PROBE), BHd) / max(day_ref, 1e-12)
    na, nm, _ = restricted_linear(ntask_enc, x_ids, 12, NSHIFT)
    m["number_task_acc"], m["number_task_margin"] = na, nm
    ns_a, ns_m, _ = restricted_linear(nsucc_enc, x_ids, 12, 1)
    m["number_succ_acc"], m["number_succ_margin"] = ns_a, ns_m
    m["day_task_acc"], m["day_task_margin"] = restricted_mod(dtask_enc, p_ids, 7, DSHIFT)
    ma, mm, _ = restricted_linear(mtask_enc, mo_ids, 12, MSHIFT)
    m["month_task_acc"], m["month_task_margin"] = ma, mm
    if wiki_ids is not None:
        lp = wiki_logp()
        kl = F.kl_div(lp.flatten(0, 1), ref_logp.flatten(0, 1),
                      reduction="none", log_target=True).sum(-1).reshape(lp.shape[:2])
        m["wiki_kl"] = float(kl.mean())
        if wiki_clean is not None and bool(wiki_clean.any()):
            m["wiki_kl_clean"] = float(kl[wiki_clean].mean())
        else:
            m["wiki_kl_clean"] = float("nan")
    return m


conds = {"T_out_all": names}
for cname, nset in conds.items():
    rows = []
    for lam in LAMBDAS:
        set_lambda(nset, lam)
        rows.append({"lam": lam, **measure()})
    set_lambda(nset, 1.0)
    torch.cuda.empty_cache()
    json.dump(rows, open(f"{OUTDIR}/{cname}.json", "w"))
    print(cname, "done; lam0:",
          json.dumps({k: round(v, 3) for k, v in rows[LAMBDAS.index(0.0)].items()
                      if not k.startswith("_")}), flush=True)

# Random controls: r0 (required), r1 (runtime permitting).
for s in range(2):
    g = torch.Generator().manual_seed(s)
    Qr, _ = torch.linalg.qr(torch.randn(HID, 6, generator=g))
    Q_t32 = {}
    R_rand, scale = {}, {}
    for n in names:
        l, kind = int(n.split("_")[1]), n.split("_")[0]
        Wp = proj_weight(l, kind)
        if Wp.device not in Q_t32:
            Q_t32[Wp.device] = Qr.float().to(Wp.device)
        Qd = Q_t32[Wp.device]
        parts = [Qd.T @ Wp[:, c0:min(c0 + 4096, Wp.shape[1])].float()
                 for c0 in range(0, Wp.shape[1], 4096)]
        R_rand[n] = torch.cat(parts, dim=1)
        scale[n] = float(R_orig[n].norm() / max(float(R_rand[n].norm()), 1e-9))
    rows = []
    for lam in LAMBDAS:
        set_lambda(names, lam, Q_t32, R_rand, scale)
        rows.append({"lam": lam, **measure()})
    set_lambda(names, 1.0, Q_t32, R_rand, scale)
    torch.cuda.empty_cache()
    json.dump(rows, open(f"{OUTDIR}/C_out_rand_r{s}.json", "w"))
    print(f"C_out_rand_r{s} done", flush=True)

print("ALL DONE", flush=True)
