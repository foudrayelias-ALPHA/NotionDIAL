"""R5: Mistral echo robustness (preregistration_robustness.md, bf03695).

Dose the L26 token-representation plane at writers 0-25; read the answer's
output code, its L26 token-code (probe transfer), and behavior.

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

MODEL = "mistralai/Mistral-7B-v0.1"
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
    MODEL, torch_dtype=torch.float16, device_map="cuda:0").eval()
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

p_ids, x_ids = tid(DAYS), tid(MONTHS)
oe = model.get_output_embeddings()
ie = model.get_input_embeddings()
tied = oe is None or oe.weight is ie.weight
W_out_rows = (ie if tied else oe).weight[p_ids].detach().float().cpu().numpy().astype(np.float64)
print("tied embeddings:", tied, flush=True)

Xc = W_out_rows - W_out_rows.mean(0)
_, S, Vt = np.linalg.svd(Xc, full_matrices=False)
B6 = np.ascontiguousarray(Vt[:6].T)                     # (HID, 6)
B6_t = torch.from_numpy(B6).half().to(DEV)
B6_t32 = torch.from_numpy(B6).float().to(DEV)

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


def proj_weight(l, kind):
    b = blocks[l]
    return (b.self_attn.o_proj if kind == "attn" else b.mlp.down_proj).weight

# ============ R5 echo robustness ============
from sklearn.linear_model import Ridge

SET_B = ["My appointment falls on {}.", "They left early on {} afternoon.",
         "Classes resume on {}.", "The concert is this {}.",
         "He was born on a {}.", "Trash pickup happens every {}.",
         "The flight departs {} night.", "Payday lands on {}."]
FRAMES = [
    ("Let's do some day of the week math. Two days after Monday is Wednesday. "
     "Two days after Friday is Sunday. Two days after {} is"),
    ("Day math: two days after Tuesday is Thursday. Two days after Saturday is "
     "Monday. Two days after {} is"),
    ("If today is Monday, in two days it will be Wednesday. If today is Friday, "
     "in two days it will be Sunday. If today is {}, in two days it will be"),
]
TOK_LAYER = 26
t7 = 2 * np.pi * np.arange(7) / 7
HdA = cloud(day_tpl, 7, TOK_LAYER)
day_tplB = positions(SET_B, DAYS, p_ids)
HdB = cloud(day_tplB, 7, TOK_LAYER)
BHA, _ = pca2(HdA)
BHB, _ = pca2(HdB)
QA, _ = np.linalg.qr(BHA); QB, _ = np.linalg.qr(BHB)
r1cos = [float(x) for x in np.linalg.svd(QA.T @ QB, compute_uv=False)]
print("R1_MISTRAL cos(A,B):", r1cos, flush=True)
probe26 = Ridge(alpha=1e-3).fit(HdA, np.stack([np.cos(t7), np.sin(t7)], 1))
want = [(i + SHIFT) % 7 for i in range(7)]
def probe_acc(A26):
    pred = probe26.predict(A26)
    ang = np.arctan2(pred[:, 1], pred[:, 0]) % (2 * np.pi)
    dec = [int(round(a / (2 * np.pi / 7))) % 7 for a in ang]
    return sum(d == w for d, w in zip(dec, want)) / 7

frames_enc = {f"f{i+1}": ctx_positions(f) for i, f in enumerate(FRAMES)}
base = {}
for k, encs in frames_enc.items():
    acc, mar = restricted(encs, SHIFT)
    base[k] = {"acc": acc, "margin": mar, "eligible": acc >= 5/7 - 1e-9}
print("baselines:", json.dumps(base), flush=True)

BH_t32 = torch.from_numpy(np.ascontiguousarray(BHA)).float().to(DEV)
names26 = [f"{k}_{l}" for l in range(TOK_LAYER) for k in ("attn", "mlp")]
R26 = {}
for n in names26:
    l, kind = int(n.split("_")[1]), n.split("_")[0]
    R26[n] = (BH_t32.T @ proj_weight(l, kind).float()).clone()
def set_tok(nset, lam):
    with torch.no_grad():
        for n in nset:
            l, kind = int(n.split("_")[1]), n.split("_")[0]
            W = proj_weight(l, kind)
            cur = BH_t32.T @ W.float()
            W += (BH_t32 @ (lam * R26[n] - cur)).half()

late = [n for n in names26 if 20 <= int(n.split("_")[1]) <= 25]
res = {"R1_mistral_cos": r1cos, "baselines": base, "conditions": {}}
for cname, nset in [("T_tok_all", names26), ("T_tok_late", late)]:
    rows = {}
    for lam in [0.0, 1.0]:
        set_tok(names26, 1.0); set_tok(nset, lam)
        row = {"probe26": probe_acc(cloud(task_enc, 7, TOK_LAYER)),
               "tok_plane_setB": plane_power(cloud(day_tplB, 7, TOK_LAYER), BHB)
                                 / max(plane_power(HdB, BHB), 1e-12)}
        for k, encs in frames_enc.items():
            if base[k]["eligible"]:
                acc, mar = restricted(encs, SHIFT)
                row[k] = {"acc": acc, "margin": mar}
        rows[lam] = row
        print(cname, "lam", lam, json.dumps(row), flush=True)
    set_tok(names26, 1.0)
    res["conditions"][cname] = rows
json.dump(res, open(f"{OUTDIR}/r5.json", "w"), indent=2)
print("ALL DONE", flush=True)
