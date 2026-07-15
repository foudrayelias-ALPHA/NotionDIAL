"""7B rung: survey -> attribution -> dosing sweep on Mistral-7B-v0.1 (Kaggle T4).

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

# ================= STAGE 1: survey (P-M1) =================
from sklearn.linear_model import Ridge

t7 = 2 * np.pi * np.arange(7) / 7
A_l = np.zeros((NL + 1, 7, HID))
Hd_l = np.zeros((NL + 1, 7, HID))
handle = blocks[NL - 1].register_forward_hook(_hook("top"))
with torch.no_grad():
    for enc, wi, pos in task_enc:
        hs = model(input_ids=enc, output_hidden_states=True).hidden_states
        for l in range(NL):
            A_l[l, wi] = hs[l][0, -1].float().cpu().numpy()
        A_l[NL, wi] = grabbed["top"][0, -1].float().cpu().numpy()
    for enc, wi, pos in day_tpl:
        hs = model(input_ids=enc, output_hidden_states=True).hidden_states
        for l in range(NL):
            Hd_l[l, wi] += hs[l][0, pos].float().cpu().numpy()
        Hd_l[NL, wi] += grabbed["top"][0, pos].float().cpu().numpy()
handle.remove()
Hd_l /= len(TEMPLATES)
want = [(i + SHIFT) % 7 for i in range(7)]
Wfull = (ie if tied else oe).weight[p_ids].detach().float().cpu()
survey = []
with torch.no_grad():
    for l in range(NL + 1):
        h = norm(torch.from_numpy(A_l[l]).half().to(DEV)).float().cpu()
        dl = h @ Wfull.T
        lens = int(sum(int(dl[i].argmax()) == want[i] for i in range(7)))
        probe = Ridge(alpha=1e-3).fit(Hd_l[l], np.stack([np.cos(t7), np.sin(t7)], 1))
        pred = probe.predict(A_l[l])
        ang = np.arctan2(pred[:, 1], pred[:, 0]) % (2 * np.pi)
        dec = [int(round(a / (2 * np.pi / 7))) % 7 for a in ang]
        pacc = int(sum(d == w for d, w in zip(dec, want)))
        survey.append({"layer": l, "lens": lens, "probe": pacc})
json.dump({"tied": tied, "rows": survey}, open(f"{OUTDIR}/survey.json", "w"))
print("SURVEY", json.dumps([r["lens"] for r in survey]),
      json.dumps([r["probe"] for r in survey]), flush=True)

# ================= STAGE 2: attribution (P-M2) =================
names = [f"{k}_{l}" for l in range(NL) for k in ("attn", "mlp")]
caps = {n: np.zeros((7, HID)) for n in names}
caps["emb"] = np.zeros((7, HID))
total = np.zeros((7, HID))
handles = [blocks[l].self_attn.register_forward_hook(_hook(f"attn_{l}"))
           for l in range(NL)]
handles += [blocks[l].mlp.register_forward_hook(_hook(f"mlp_{l}"))
            for l in range(NL)]
handles.append(blocks[NL - 1].register_forward_hook(_hook("top")))
with torch.no_grad():
    for enc, wi, pos in task_enc:
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

# ================= STAGE 3: dosing sweep (P-M3..P-M5) =================
def proj_weight(l, kind):
    b = blocks[l]
    return (b.self_attn.o_proj if kind == "attn" else b.mlp.down_proj).weight

# per-module original rank-6 content R = B6^T W (fp32, exact target math)
R_orig = {}
for l in range(NL):
    for kind in ("attn", "mlp"):
        W = proj_weight(l, kind)
        R_orig[f"{kind}_{l}"] = (B6_t32.T @ W.float()).clone()   # (6, in)

def set_lambda(names_set, lam, Q_t32=None, R_rand=None, scale=None):
    """Exact re-pin: subspace content of each module set to its closed form."""
    with torch.no_grad():
        for n in names_set:
            l, kind = int(n.split("_")[1]), n.split("_")[0]
            W = proj_weight(l, kind)
            if Q_t32 is None:  # targeted plane condition
                target = lam * R_orig[n]
                cur = B6_t32.T @ W.float()
                W += (B6_t32 @ (target - cur)).half()
            else:              # random control, norm-matched scale s
                target = (1.0 - (1.0 - lam) * scale[n]) * R_rand[n]
                cur = Q_t32.T @ W.float()
                W += (Q_t32 @ (target - cur)).half()

wiki_ids = None
try:
    from datasets import load_dataset
    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n".join(t for t in wiki["text"] if t.strip())[:20000]
    wiki_ids = tok(text, return_tensors="pt").input_ids[0][: 8 * 64].reshape(8, 64).to(DEV)
except Exception as e:
    print("wiki unavailable:", e, flush=True)

Hd19 = cloud(day_tpl, 7, PROBE)
BH, _ = pca2(Hd19)
day_ref = plane_power(Hd19, BH)
Hm19 = cloud(month_tpl, 12, PROBE)
BHm, _ = pca2(Hm19)
month_ref = plane_power(Hm19, BHm)
A_ref_cloud = cloud(task_enc, 7, NL)
A_ref = plane_power(A_ref_cloud, B6)
if wiki_ids is not None:
    with torch.no_grad():
        ref_logp = F.log_softmax(model(input_ids=wiki_ids).logits.float(), -1).cpu()

def measure():
    m = {}
    Acl = cloud(task_enc, 7, NL)
    m["A_B6_power"] = plane_power(Acl, B6) / max(A_ref, 1e-12)
    m["_A_cloud_B6"] = [[round(float(v), 4) for v in row] for row in
                        ((Acl - Acl.mean(0)) @ B6)[:, :2]]
    m["day_plane_power"] = plane_power(cloud(day_tpl, 7, PROBE), BH) / max(day_ref, 1e-12)
    m["month_plane_power"] = plane_power(cloud(month_tpl, 12, PROBE), BHm) / max(month_ref, 1e-12)
    m["task_acc"], m["task_margin"] = restricted(task_enc, SHIFT)
    m["succ_acc"], m["succ_margin"] = restricted(succ_enc, 1)
    m["after1_acc"], m["after1_margin"] = restricted(after1_enc, 1)
    if wiki_ids is not None:
        with torch.no_grad():
            lp = F.log_softmax(model(input_ids=wiki_ids).logits.float(), -1).cpu()
        m["wiki_kl"] = float(F.kl_div(lp.flatten(0, 1), ref_logp.flatten(0, 1),
                                      reduction="batchmean", log_target=True))
    return m

attn_names = [n for n in names if n.startswith("attn")]
mlp_names = [n for n in names if n.startswith("mlp")]
conds = {"T_out_top": [att["top_writer"]], "T_out_allattn": attn_names,
         "T_out_allmlp": mlp_names, "T_out_all": names}
for cname, nset in conds.items():
    rows = []
    for lam in LAMBDAS:
        set_lambda(nset, lam)
        rows.append({"lam": lam, **measure()})
    set_lambda(nset, 1.0)
    json.dump(rows, open(f"{OUTDIR}/{cname}.json", "w"))
    print(cname, "done; lam0:",
          json.dumps({k: round(v, 3) for k, v in rows[LAMBDAS.index(0.0)].items()
                      if not k.startswith("_")}), flush=True)

for s in range(3):
    g = torch.Generator().manual_seed(s)
    Qr, _ = torch.linalg.qr(torch.randn(HID, 6, generator=g))
    Q_t32 = Qr.float().to(DEV)
    R_rand, scale = {}, {}
    for n in names:
        l, kind = int(n.split("_")[1]), n.split("_")[0]
        W = proj_weight(l, kind).float()
        R_rand[n] = (Q_t32.T @ W).clone()
        scale[n] = float(R_orig[n].norm() / max(float(R_rand[n].norm()), 1e-9))
    rows = []
    for lam in LAMBDAS:
        set_lambda(names, lam, Q_t32, R_rand, scale)
        rows.append({"lam": lam, **measure()})
    set_lambda(names, 1.0, Q_t32, R_rand, scale)
    json.dump(rows, open(f"{OUTDIR}/C_out_rand_r{s}.json", "w"))
    print(f"C_out_rand_r{s} done", flush=True)

print("ALL DONE", flush=True)
