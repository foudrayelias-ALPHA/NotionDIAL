#!/usr/bin/env python3
"""Build the single-page lambda-dial demo from committed artifacts only.

Five model panels (124M -> 7B), one global dial
W(lambda) = W + (lambda-1)*M, with a THREE-state concept switch: the same panels
swap between the DAYS experiment, the MONTHS experiment (the full months battery,
prereg 5adc074), and the NUMBERS experiment (prereg 5745ec7; a number line, open
in numeric order rather than a closed ring, since the numbers do not wrap).
Every embedded number is a measured forward pass at one of the 26
preregistered lambda doses. This builder does NO model computation: it reads the
committed artifact JSONs, validates them against an honesty gate, slims each row
to (lam + the panel's metric keys + the cloud field), keeping the metric keys
for BOTH target and control rows so the page can show the target-vs-control
contrast, and injects the result into demo/template.html at the ___DATA___
placeholder.

Every panel carries a spared neighbor concept riding the same edited weights;
that, plus the behavior readouts, is the specificity evidence. Where a
preregistered criterion failed, the panel states it plainly (months P-MS4 and
P-MT-KLC on GPT-2; behavioral-collapse misses on Llama/Qwen-1.5B/Mistral months;
specificity miss on Qwen-7B months).

Honesty gate: for every panel VARIANT (days, months, and numbers where present)
the cloud field must be present in all 26 rows with that variant's own exact
point count (12 for the GPT-2 number token line, 10 for the four answer-manifold
number lines, and the per-panel counts for days/months), all finite [x, y] pairs,
in both the target and the control artifact. The default build requires every
variant of every panel; a failure is hard and names the file and field. Pass
--partial to build only what validates (loud warning; a model panel keeps the
sides that validate).

Usage:
  python3 demo/build_demo.py            # requires all panels, all three concepts
  python3 demo/build_demo.py --partial  # build whatever validates (dev)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../spd-manifold-clock/demo
REPO = HERE.parent                               # .../spd-manifold-clock
ART = REPO / "artifacts"

CANONICAL_LAMS = [-0.5, -0.25, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
                  0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
                  1.0, 1.1, 1.25, 1.5]

DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
NUMBER_LABELS = [str(i) for i in range(1, 13)]   # tokens one..twelve (12)
NUMBER_ANS_LABELS = [str(i) for i in range(1, 11)]  # 2-hop answer items one..ten (10)


def ring(labels):
    return {"kind": "ring", "closed": True, "hue": "wheel",
            "labels": labels, "n_points": len(labels)}


def line(labels):
    """Open polyline in numeric order, sequential coloring (the numbers side).
    Not a closed ring: the number line does not wrap."""
    return {"kind": "line", "closed": False, "hue": "seq",
            "labels": labels, "n_points": len(labels)}


# Verdict captions on the months side. Every number below was checked against
# the artifact rows at lambda=0 before being written here; the same rows are in
# the payload, so the page shows them next to the caption.
NOTE_GPT2_MO = (
    "Preregistered specificity criterion P-MS4 FAILED (experiment scored 4/5): "
    "at λ=0 the targeted edit costs more wiki KL (0.265) than the random control "
    "(0.089). The clean-context cell (P-MT-KLC) confirms it: the wiki batch "
    "contains zero month tokens, so clean KL equals raw KL and the specificity "
    "miss is real for this token-plane edit, not a wikitext artifact."
)
NOTE_LLAMA_MO = (
    "Preregistered behavioral-collapse criterion failed as frozen: the month "
    "answer manifold is killed exactly, but month 1-hop accuracy only falls to "
    "6/12. Months stand less on this manifold than days do."
)
NOTE_QWEN15_MO = (
    "Preregistered behavioral-collapse criterion failed as frozen: the month "
    "answer manifold is killed exactly, but month 2-hop accuracy only falls to "
    "5/12. Months stand less on this manifold than days do."
)
NOTE_MISTRAL_MO = (
    "Preregistered behavioral-collapse criterion failed as frozen: the month "
    "answer manifold is killed exactly, but month task accuracy only falls to "
    "7/12. Months stand less on this manifold than days do."
)
NOTE_QWEN7B_MO = (
    "Full behavioral collapse like days (12/12 to 2/12, monotone, ρ = 1.0). The "
    "preregistered specificity criterion failed instead: targeted clean KL 0.0185 "
    "vs 0.0144 for the r0 random control (second control seed 0.0209)."
)

# Verdict captions on the NUMBERS side (prereg 5745ec7). The clean-context
# specificity criterion failed at every anatomy (targeted clean KL > matched
# random control in all five models); that tab-level fact is stated once near
# the switch. Per-panel notes carry the anatomy-specific finding.
NOTE_GPT2_NU = (
    "Full counting collapse riding the dose: run-up counting falls 9/9 to 0/9. "
    "Day/month sparing FAILED here: at λ=0 the same edit drags the day plane to "
    "0.84 and the month plane to 0.70. The day, month, and number planes overlap "
    "at layer 8, so this token-plane edit is not number-specific."
)
NOTE_LLAMA_NU = (
    "Exact kill of the number answer manifold, but only partial behavioral "
    "collapse: number 2-hop accuracy falls to 4/10. Numbers stand partly on this "
    "manifold. Clean-context specificity failed (targeted clean KL 0.080 vs 0.040 "
    "for the random control)."
)
NOTE_QWEN15_NU = (
    "Exact kill of the number answer manifold; number 2-hop accuracy falls to "
    "2/10 (floor met). Disclosed entanglement: the day answer manifold shares "
    "subspace with the number answer manifold (day token plane holds at 0.93, "
    "carrying the sparing clause). Clean-context specificity failed (targeted "
    "clean KL 0.254 vs 0.168 for the random control)."
)
NOTE_MISTRAL_NU = (
    "Exact kill of the number answer manifold, but the weakest behavioral "
    "collapse: number task accuracy only falls to 7/10, with day 10/10 and month "
    "9/10 fully spared. Clean-context specificity failed (targeted clean KL 0.040 "
    "vs 0.012 for the random control)."
)
NOTE_QWEN7B_NU = (
    "The cleanest full co-collapse: number task accuracy falls 10/10 to 0/10 "
    "(margin −4.605, ρ = 1.0) as the answer manifold is killed exactly, with day "
    "10/10 and month 8/10 spared. Clean-context specificity failed instead "
    "(targeted clean KL 0.691 vs 0.015 for the random control): number words "
    "pervade ordinary text, so the representation is entangled with general "
    "computation."
)

# Panel specs. section 1 = five model scales with a days/months concept switch
# ("alt" is the months side). kind: "ring" (closed 2-D polygon),
# "line" (open 2-D polyline).
PANELS = [
    # ---- section 1: five model scales, days <-> months -------------------
    {
        "section": 1, "name": "GPT-2", "size": "124M",
        **ring(DAY_LABELS),
        "sub": "constructed day ring, layer 8",
        "cloud_field": "_H_cloud_B2",
        "target": "writers_gpt2/sweep/T_wo_all.json",
        "control": "writers_gpt2/sweep/C_wo_rand_r0.json",
        "metrics": [
            ["day_plane_power", "days concept"],
            ["succ_acc", "days task"],
            ["month_plane_power", "months concept"],
        ],
        "prov": "artifacts/writers_gpt2/sweep/T_wo_all.json · prereg abb07c0",
        "alt": {
            **ring(MONTH_LABELS),
            "sub": "months as target, days spared (the symmetric swap), layer 8",
            "cloud_field": "_H_cloud_B2",
            "target": "monthswap_gpt2/sweep/T_mo_all.json",
            "control": "monthswap_gpt2/sweep/C_mo_rand_r0.json",
            "metrics": [
                ["month_plane_power", "months concept"],
                ["day_succ_acc", "days task"],
                ["day_plane_power", "days concept"],
            ],
            "note": NOTE_GPT2_MO,
            "prov": "artifacts/monthswap_gpt2/sweep/T_mo_all.json · prereg 349a99f + 5adc074",
        },
        "num": {
            **line(NUMBER_LABELS),
            "sub": "number token line one..twelve, layer 8 (token-plane edit)",
            "cloud_field": "_H_cloud_B2",
            "target": "numberswap_gpt2/sweep/T_nu_all.json",
            "control": "numberswap_gpt2/sweep/C_nu_rand_r0.json",
            "metrics": [
                ["number_plane_power", "numbers concept"],
                ["count_acc", "counting"],
                ["day_plane_power", "days concept"],
            ],
            "note": NOTE_GPT2_NU,
            "prov": "artifacts/numberswap_gpt2/sweep/T_nu_all.json · prereg 5745ec7",
        },
    },
    {
        "section": 1, "name": "Llama-3.2-1B", "size": "1B",
        **ring(DAY_LABELS),
        "sub": "computed answer manifold (“day after X”)",
        "cloud_field": "_A_cloud_B6",
        "target": "contextual_llama/sweep_out/T_out_all.json",
        "control": "contextual_llama/sweep_out/C_out_rand_r0.json",
        "metrics": [
            ["A_B6_power", "days concept"],
            ["ctx_acc", "days task"],
            ["month_plane_power", "months concept"],
        ],
        "prov": "artifacts/contextual_llama/sweep_out/T_out_all.json · prereg 80b0793+529dbd4",
        "alt": {
            **ring(MONTH_LABELS),
            "sub": "computed month answer manifold (“month after X”)",
            "cloud_field": "_A_cloud_B6",
            "target": "monthctx_llama/sweep_out/T_out_all.json",
            "control": "monthctx_llama/sweep_out/C_out_rand_r0.json",
            "metrics": [
                ["A_mo_B6_power", "months concept"],
                ["month_1hop_acc", "months task"],
                ["day_tokplane_power", "days concept"],
            ],
            "note": NOTE_LLAMA_MO,
            "prov": "artifacts/monthctx_llama/sweep_out/T_out_all.json · prereg 5adc074",
        },
        "num": {
            **line(NUMBER_ANS_LABELS),
            "sub": "computed number answer manifold (2-hop, ten non-wrapping items)",
            "cloud_field": "_A_cloud_B6",
            "target": "numberctx_llama/sweep/T_out_all.json",
            "control": "numberctx_llama/sweep/C_out_rand_r0.json",
            "metrics": [
                ["A_nu_B6_power", "numbers concept"],
                ["number_2hop_acc", "numbers task"],
                ["day_tokplane_power", "days concept"],
            ],
            "note": NOTE_LLAMA_NU,
            "prov": "artifacts/numberctx_llama/sweep/T_out_all.json · prereg 5745ec7",
        },
    },
    {
        "section": 1, "name": "Qwen2.5-1.5B", "size": "1.5B",
        **ring(DAY_LABELS),
        "sub": "2-hop day arithmetic answer",
        "cloud_field": "_A_cloud_B6",
        "target": "qwen2hop/sweep/T_out_all.json",
        "control": "qwen2hop/sweep/C_out_rand_r0.json",
        "metrics": [
            ["A_B6_power", "days concept"],
            ["few2_acc", "days task"],
            ["month_plane_power", "months concept"],
        ],
        "prov": "artifacts/qwen2hop/sweep/T_out_all.json · prereg 193f444",
        "alt": {
            **ring(MONTH_LABELS),
            "sub": "2-hop month arithmetic answer",
            "cloud_field": "_A_cloud_B6",
            "target": "month2hop_qwen/sweep/T_out_all.json",
            "control": "month2hop_qwen/sweep/C_out_rand_r0.json",
            "metrics": [
                ["A_mo_B6_power", "months concept"],
                ["month_2hop_acc", "months task"],
                ["day_tokplane_power", "days concept"],
            ],
            "note": NOTE_QWEN15_MO,
            "prov": "artifacts/month2hop_qwen/sweep/T_out_all.json · prereg 5adc074",
        },
        "num": {
            **line(NUMBER_ANS_LABELS),
            "sub": "computed number answer manifold (2-hop, ten non-wrapping items)",
            "cloud_field": "_A_cloud_B6",
            "target": "number2hop_qwen/sweep/T_out_all.json",
            "control": "number2hop_qwen/sweep/C_out_rand_r0.json",
            "metrics": [
                ["A_nu_B6_power", "numbers concept"],
                ["number_2hop_acc", "numbers task"],
                ["day_tokplane_power", "days concept"],
            ],
            "note": NOTE_QWEN15_NU,
            "prov": "artifacts/number2hop_qwen/sweep/T_out_all.json · prereg 5745ec7",
        },
    },
    {
        "section": 1, "name": "Mistral-7B", "size": "7B",
        **ring(DAY_LABELS),
        "sub": "2-hop at scale",
        "cloud_field": "_A_cloud_B6",
        "target": "mistral7b/T_out_all.json",
        "control": "mistral7b/C_out_rand_r0.json",
        "metrics": [
            ["A_B6_power", "days concept"],
            ["task_acc", "days task"],
            ["month_plane_power", "months concept"],
        ],
        "prov": "artifacts/mistral7b/T_out_all.json · prereg a95a423",
        "alt": {
            **ring(MONTH_LABELS),
            "sub": "months 2-hop at scale",
            "cloud_field": "_A_cloud_B6",
            "target": "mistral7b_months/T_out_all.json",
            "control": "mistral7b_months/C_out_rand_r0.json",
            "metrics": [
                ["A_B6_power", "months concept"],
                ["month_task_acc", "months task"],
                ["day_plane_power", "days concept"],
            ],
            "note": NOTE_MISTRAL_MO,
            "prov": "artifacts/mistral7b_months/T_out_all.json · prereg 5adc074",
        },
        "num": {
            **line(NUMBER_ANS_LABELS),
            "sub": "number answer manifold at scale, ten non-wrapping items",
            "cloud_field": "_A_cloud_B6",
            "target": "mistral7b_numbers/T_out_all.json",
            "control": "mistral7b_numbers/C_out_rand_r0.json",
            "metrics": [
                ["A_B6_power", "numbers concept"],
                ["number_task_acc", "numbers task"],
                ["day_task_acc", "days task"],
            ],
            "note": NOTE_MISTRAL_NU,
            "prov": "artifacts/mistral7b_numbers/T_out_all.json · prereg 5745ec7",
        },
    },
    {
        "section": 1, "name": "Qwen2.5-7B", "size": "7B",
        **ring(DAY_LABELS),
        "sub": "2-hop, second 7B family",
        "cloud_field": "_A_cloud_B6",
        "target": "qwen7b/T_out_all.json",
        "control": "qwen7b/C_out_rand_r0.json",
        "metrics": [
            ["A_B6_power", "days concept"],
            ["task_acc", "days task"],
            ["month_plane_power", "months concept"],
        ],
        "prov": "artifacts/qwen7b/T_out_all.json · prereg 1190237",
        "alt": {
            **ring(MONTH_LABELS),
            "sub": "months 2-hop, second 7B family",
            "cloud_field": "_A_cloud_B6",
            "target": "qwen7b_months/T_out_all.json",
            "control": "qwen7b_months/C_out_rand_r0.json",
            "metrics": [
                ["A_B6_power", "months concept"],
                ["month_task_acc", "months task"],
                ["day_plane_power", "days concept"],
            ],
            "note": NOTE_QWEN7B_MO,
            "prov": "artifacts/qwen7b_months/T_out_all.json · prereg 5adc074",
        },
        "num": {
            **line(NUMBER_ANS_LABELS),
            "sub": "number answer manifold, second 7B family, ten items",
            "cloud_field": "_A_cloud_B6",
            "target": "qwen7b_numbers/T_out_all.json",
            "control": "qwen7b_numbers/C_out_rand_r0.json",
            "metrics": [
                ["A_B6_power", "numbers concept"],
                ["number_task_acc", "numbers task"],
                ["day_task_acc", "days task"],
            ],
            "note": NOTE_QWEN7B_NU,
            "prov": "artifacts/qwen7b_numbers/T_out_all.json · prereg 5745ec7",
        },
    },
]


class GateError(Exception):
    pass


def _load(rel):
    path = ART / rel
    if not path.exists():
        raise GateError(f"artifact not found: {path}")
    with open(path) as f:
        return json.load(f)


def _is_finite_num(x):
    return isinstance(x, (int, float)) and x == x and abs(x) != float("inf")


def _finite_point(p):
    return (isinstance(p, (list, tuple)) and len(p) == 2
            and all(_is_finite_num(x) for x in p))


def _validate_cloud(rows, field, rel, n_points):
    """Every one of the 26 rows must carry `field` with exactly n_points finite
    [x, y] pairs."""
    if len(rows) != 26:
        raise GateError(f"{rel}: expected 26 rows, found {len(rows)}")
    for i, r in enumerate(rows):
        if field not in r or r[field] is None:
            raise GateError(f"{rel}: cloud field '{field}' missing in row {i} (lam={r.get('lam')})")
        cloud = r[field]
        if not isinstance(cloud, list) or len(cloud) != n_points:
            raise GateError(f"{rel}: field '{field}' in row {i} (lam={r.get('lam')}) "
                            f"is not {n_points} points (got {len(cloud) if isinstance(cloud, list) else type(cloud).__name__})")
        for j, p in enumerate(cloud):
            if not _finite_point(p):
                raise GateError(f"{rel}: field '{field}' row {i} point {j} is not a finite [x,y]: {p!r}")


def _slim(rows, field, metric_keys):
    """Keep only lam, the panel's metric keys, and the cloud field per row."""
    out = []
    for r in rows:
        cloud = [[round(float(x), 4) for x in p] for p in r[field]]
        row = {"lam": round(float(r["lam"]), 4), "cloud": cloud}
        for k in metric_keys:
            row[k] = r.get(k)
        out.append(row)
    out.sort(key=lambda x: x["lam"])
    return out


def build_variant(spec):
    """Validate + slim one variant (one concept side of one panel)."""
    field = spec["cloud_field"]
    n = spec["n_points"]
    metric_keys = [k for k, _ in spec["metrics"]]

    t_rows = _load(spec["target"])
    _validate_cloud(t_rows, field, spec["target"], n)

    c_rows = _load(spec["control"])
    _validate_cloud(c_rows, field, spec["control"], n)

    lams = sorted(round(float(r["lam"]), 4) for r in t_rows)
    if lams != CANONICAL_LAMS:
        raise GateError(f"{spec['target']}: lam grid does not match the canonical 26 doses")

    variant = {
        "kind": spec["kind"],
        "closed": spec["closed"],
        "hue": spec["hue"],
        "sub": spec["sub"],
        "prov": spec["prov"],
        "metrics": spec["metrics"],
        "labels": spec["labels"],
        "target": _slim(t_rows, field, metric_keys),
        "control": _slim(c_rows, field, metric_keys),
    }
    if spec.get("note"):
        variant["note"] = spec["note"]
    return variant


def main():
    partial = "--partial" in sys.argv[1:]

    built, excluded = [], []
    for spec in PANELS:
        try:
            panel = {"section": spec["section"], "name": spec["name"], "size": spec["size"]}
            panel.update(build_variant(spec))          # days / single concept
        except GateError as e:
            if not partial:
                sys.stderr.write(
                    "\nBUILD FAILED (honesty gate).\n"
                    f"  Panel : {spec['name']} ({spec['size']})\n"
                    f"  Field : {spec['cloud_field']}\n"
                    f"  Reason: {e}\n\n"
                    "The default build requires every panel (both concept states where present).\n"
                    "Use --partial to build only what validates (dev only).\n"
                )
                sys.exit(1)
            excluded.append((spec["name"], str(e)))
            continue
        for side_key, side_name in (("alt", "months side"), ("num", "numbers side")):
            if not spec.get(side_key):
                continue
            try:
                panel[side_key] = build_variant(spec[side_key])
            except GateError as e:
                if not partial:
                    sys.stderr.write(
                        f"\nBUILD FAILED (honesty gate, {side_name}).\n"
                        f"  Panel : {spec['name']} ({spec['size']})\n"
                        f"  Field : {spec[side_key]['cloud_field']}\n"
                        f"  Reason: {e}\n\n"
                        "The default build requires every panel (all concept states where present).\n"
                        "Use --partial to build only what validates (dev only).\n"
                    )
                    sys.exit(1)
                excluded.append((f"{spec['name']} ({side_name})", str(e)))
        built.append(panel)

    if partial and excluded:
        sys.stderr.write("\n" + "!" * 68 + "\n")
        sys.stderr.write("WARNING: --partial build. The following were EXCLUDED because\n"
                         "their measured cloud data does not yet validate:\n")
        for name, reason in excluded:
            sys.stderr.write(f"  - {name}: {reason}\n")
        sys.stderr.write("This build is INCOMPLETE and must not be shipped as final.\n")
        sys.stderr.write("!" * 68 + "\n\n")

    if not built:
        sys.stderr.write("No panels validated; nothing to build.\n")
        sys.exit(1)

    data = {"lams": CANONICAL_LAMS, "panels": built}

    template = (HERE / "template.html").read_text()
    if "___DATA___" not in template:
        raise SystemExit(f"placeholder ___DATA___ not found in {HERE / 'template.html'}")
    html = template.replace("___DATA___", json.dumps(data, separators=(",", ":")))

    out_path = HERE / "index.html"
    out_path.write_text(html)
    n_variants = sum(1 + (1 if p.get("alt") else 0) + (1 if p.get("num") else 0)
                     for p in built)
    print(f"demo built: {len(html)} bytes, {len(built)}/{len(PANELS)} panels "
          f"({n_variants} concept variants) -> {out_path}")
    print("panels:", ", ".join(
        p["name"] + "".join(s for s, k in (("+months", "alt"), ("+numbers", "num"))
                            if p.get(k))
        for p in built))


if __name__ == "__main__":
    main()
