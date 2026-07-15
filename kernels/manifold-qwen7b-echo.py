"""Qwen2.5-7B ECHO test (preregistration_qwen7b.md addendum, 88c3469).
Chunk-safe rank-2 edits on a fully GPU-resident model (P100).

Self-contained port of the spd-manifold-clock pipeline (preregistration addendum
frozen locally before this kernel ran). Deviations from the local pipeline, all
documented in decisions.md: fp16 forward (fp64 numpy edit math unchanged);
in-place rank-6 edits with exact re-pinning per lambda (no weight clones: at
each lambda the edited-subspace content is set exactly to its closed form);
per-lambda answer clouds saved for the demo's geometry panel.
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
TEMPLATES = ["The meeting is scheduled for {}.", "She will arrive on {}.",
             "Everything closed last {} evening.", "I always go swimming on {}.",
             "The deadline is next {}.", "It happened one {} morning.",
             "We usually rest on {}.", "The store reopens on {}."]
TASK_TPL = None  # set from frozen prereg choice injected below
SHIFT = None
FEWSHOT = ("Let's do some day of the week math. Two days after Monday is "
           "Wednesday. Two days after Friday is Sunday. Two days after {} is")
AFTER1 = "The day after {} is"
SUCC = "If today is {}, then tomorrow is"
LAMBDAS = sorted({-0.5, -0.25} | {round(x, 3) for x in np.linspace(0, 1, 21)}
                 | {1.1, 1.25, 1.5})
OUTDIR = "/kaggle/working"

# ---- frozen task choice (from the screen, per prereg addendum) ----
TASK_TPL, SHIFT = FEWSHOT, 2  # overridden only if prereg says otherwise

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map={"": 0}).eval()
assert not any(p.is_meta for p in model.parameters()), "meta tensors present"
print("device map:", getattr(model, "hf_device_map", "single-device {'': 0}"), flush=True)
model.requires_grad_(False)
DEV = "cuda:0"  # input device; layers are sharded, accelerate dispatches
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

p_ids, x_ids = tid(DAYS), tid(MONTHS)
oe = model.get_output_embeddings()
ie = model.get_input_embeddings()
tied = oe is None or oe.weight is ie.weight
W_out_rows = (ie if tied else oe).weight[p_ids].detach().float().cpu().numpy().astype(np.float64)
print("tied embeddings:", tied, flush=True)

Xc = W_out_rows - W_out_rows.mean(0)
_, S, Vt = np.linalg.svd(Xc, full_matrices=False)
B6 = np.ascontiguousarray(Vt[:6].T)                     # (HID, 6)
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

def ctx_positions(tpl):
    return [(tok(tpl.format(w), return_tensors="pt").input_ids.to(DEV), wi, -1)
            for wi, w in enumerate(DAYS)]

day_tpl = positions(TEMPLATES, DAYS, p_ids)
month_tpl = positions(TEMPLATES, MONTHS, x_ids)
task_enc = ctx_positions(TASK_TPL)
succ_enc = ctx_positions(SUCC)
after1_enc = ctx_positions(AFTER1)

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
def restricted(encs, shift):
    margins, correct = [], 0
    for i, (enc, wi, pos) in enumerate(encs):
        dl = model(input_ids=enc).logits[0, -1][p_ids].float()
        want = (i + shift) % 7
        margins.append(float(dl[want] - dl[torch.arange(7) != want].max()))
        correct += int(dl.argmax()) == want
    return correct / 7, float(np.mean(margins))

def pca2(H):
    Hc = H - H.mean(0)
    _, s, vt = np.linalg.svd(Hc, full_matrices=False)
    return vt[:2].T, float((s[:2] ** 2).sum() / (s**2).sum())

def plane_power(H, B):
    Hc = H - H.mean(0)
    return float(((Hc @ B) ** 2).sum())


def proj_weight(l, kind):
    b = blocks[l]
    return (b.self_attn.o_proj if kind == "attn" else b.mlp.down_proj).weight

wiki_ids = None
try:
    from datasets import load_dataset
    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n".join(t for t in wiki["text"] if t.strip())[:20000]
    wiki_ids = tok(text, return_tensors="pt").input_ids[0][: 8 * 64].reshape(8, 64).to(DEV)
except Exception as e:
    print("wiki unavailable:", e, flush=True)

def wiki_logp():
    outs = []
    with torch.no_grad():
        for r in range(wiki_ids.shape[0]):
            lg = model(input_ids=wiki_ids[r:r + 1]).logits.float()
            outs.append(F.log_softmax(lg, -1).cpu())
    return torch.cat(outs, 0)

if wiki_ids is not None:
    ref_logp = wiki_logp()

# ================= echo test (P-2Q6) =================
from sklearn.linear_model import Ridge

TOK_LAYER = 26
t7 = 2 * np.pi * np.arange(7) / 7
HdT = cloud(day_tpl, 7, TOK_LAYER)
BH, _sh = pca2(HdT)
tok_ref = plane_power(HdT, BH)
probe26 = Ridge(alpha=1e-3).fit(HdT, np.stack([np.cos(t7), np.sin(t7)], 1))
want = [(i + SHIFT) % 7 for i in range(7)]

def probe_acc(A26):
    pred = probe26.predict(A26)
    ang = np.arctan2(pred[:, 1], pred[:, 0]) % (2 * np.pi)
    dec = [int(round(a / (2 * np.pi / 7))) % 7 for a in ang]
    return sum(d == w for d, w in zip(dec, want)) / 7

HmT = cloud(month_tpl, 12, TOK_LAYER)
BHm, _ = pca2(HmT)
month_ref = plane_power(HmT, BHm)

def B2_on_dev(B, dev, cache={}):
    key = (id(B), dev)
    if key not in cache:
        cache[key] = torch.from_numpy(np.ascontiguousarray(B)).float().to(dev)
    return cache[key]

names26 = [f"{k}_{l}" for l in range(TOK_LAYER) for k in ("attn", "mlp")]
R26 = {}
for n in names26:
    l, kind = int(n.split("_")[1]), n.split("_")[0]
    W = proj_weight(l, kind)
    Bd = B2_on_dev(BH, W.device)
    parts = [Bd.T @ W[:, c0:min(c0 + 4096, W.shape[1])].float()
             for c0 in range(0, W.shape[1], 4096)]
    R26[n] = torch.cat(parts, dim=1)

def set_tok(nset, lam, Bmat=None, R=None, scale=None):
    Bnp = BH if Bmat is None else Bmat
    Rd = R26 if R is None else R
    with torch.no_grad():
        for n in nset:
            l, kind = int(n.split("_")[1]), n.split("_")[0]
            W = proj_weight(l, kind)
            Bd = B2_on_dev(Bnp, W.device)
            coef = lam if scale is None else (1.0 - (1.0 - lam) * scale[n])
            target = coef * Rd[n]
            for c0 in range(0, W.shape[1], 4096):
                c1 = min(c0 + 4096, W.shape[1])
                cur = Bd.T @ W[:, c0:c1].float()
                W[:, c0:c1] += (Bd @ (target[:, c0:c1] - cur)).half()

def measure_echo():
    m = {}
    m["tok_plane_power"] = plane_power(cloud(day_tpl, 7, TOK_LAYER), BH) / max(tok_ref, 1e-12)
    m["answer_probe26_acc"] = probe_acc(cloud(task_enc, 7, TOK_LAYER))
    m["A_B6_power"] = plane_power(cloud(task_enc, 7, NL), B6) / max(A_top_ref, 1e-12)
    m["month_plane_power"] = plane_power(cloud(month_tpl, 12, TOK_LAYER), BHm) / max(month_ref, 1e-12)
    m["task_acc"], m["task_margin"] = restricted(task_enc, SHIFT)
    m["succ_acc"], m["succ_margin"] = restricted(succ_enc, 1)
    if wiki_ids is not None:
        lp = wiki_logp()
        m["wiki_kl"] = float(F.kl_div(lp.flatten(0, 1), ref_logp.flatten(0, 1),
                                      reduction="batchmean", log_target=True))
    return m

A_top_ref = plane_power(cloud(task_enc, 7, NL), B6)
late = [n for n in names26 if 22 <= int(n.split("_")[1]) <= 25]
for cname, nset in [("T_tok_all", names26), ("T_tok_late", late)]:
    rows = []
    for lam in LAMBDAS:
        set_tok(names26, 1.0)
        set_tok(nset, lam)
        rows.append({"lam": lam, **measure_echo()})
    set_tok(names26, 1.0)
    torch.cuda.empty_cache()
    json.dump(rows, open(f"{OUTDIR}/{cname}.json", "w"))
    print(cname, "done; lam0:", json.dumps({k: round(v, 3) for k, v in rows[LAMBDAS.index(0.0)].items()}), flush=True)

g = torch.Generator().manual_seed(0)
Qr, _ = torch.linalg.qr(torch.randn(HID, 2, generator=g))
Qnp = Qr.numpy().astype(np.float64)
R_rand, scale = {}, {}
for n in names26:
    l, kind = int(n.split("_")[1]), n.split("_")[0]
    W = proj_weight(l, kind)
    Qd = B2_on_dev(Qnp, W.device)
    parts = [Qd.T @ W[:, c0:min(c0 + 4096, W.shape[1])].float()
             for c0 in range(0, W.shape[1], 4096)]
    R_rand[n] = torch.cat(parts, dim=1)
    scale[n] = float(R26[n].norm() / max(float(R_rand[n].norm()), 1e-9))
rows = []
for lam in LAMBDAS:
    set_tok(names26, 1.0)
    set_tok(names26, lam, Qnp, R_rand, scale)
    rows.append({"lam": lam, **measure_echo()})
set_tok(names26, 1.0, Qnp, R_rand, scale)
set_tok(names26, 1.0)
json.dump(rows, open(f"{OUTDIR}/C_tok_rand_r0.json", "w"))
print("C_tok_rand_r0 done", flush=True)
print("ALL DONE", flush=True)
