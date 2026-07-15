# NotionDIAL

**Dosed Interventions Along λ.** A continuous weight-space dial for concept
manifolds in language models: pick a concept, discover its subspace with no
geometric priors, locate the weights that write it, and turn it. W(λ) = W +
(λ−1)M over a fixed 26-dose grid; the manifold's geometry in activation
space is the dependent variable, read at every dose through instruments
frozen at λ=1, with behavior and collateral recorded alongside.

Four properties define the instrument: **exact** (λ=0 removes the targeted
subspace to machine precision, at oracle parity where ground truth exists),
**continuous** (dose-response curves, not on/off ablations), **specific**
(matched-norm random controls, cross-concept sparing, collateral bounds at
every dose), and **portable** (any Hugging Face causal LM; laptop to 1.5B,
free-tier cloud GPU at 7B). Demonstrated on seven models across six
architecture families (124M to 7B), three token-anchored concepts (days,
months, numbers) and one abstract axis (valence), under a preregistration
discipline of 20 freezes. Roughly half of the findings arrived as
preregistered failures.

## What's in this repository

This is the public demo-and-code tree. The paper, measured-data artifacts,
preregistration freezes, and operational logs are maintained separately and
are **not** included here.

- **Live demo:** https://foudrayelias-alpha.github.io/NotionDIAL/ — one λ
  slider, a days | months | numbers concept switch across five models,
  placebo overlays, and every point a real forward pass. Served by GitHub
  Pages from the root `index.html`.
- **Demo source (`demo/`):** `index.html` is the self-contained, pre-built
  page with every measured value baked in; the copy at the repo root is what
  Pages serves. `build_demo.py` regenerates the page from measured-data
  artifacts under a hard honesty gate (missing data fails the build) — those
  artifacts are not shipped here, so the committed page is the reproducible
  record. `template.html` is its scaffold.
- **Library (`clocklib/`):** model loading, prior-free subspace discovery,
  frozen-basis metrics, edit families, and the dose-sweep engine.
- **Experiment scripts (`scripts/00…31`):** numbered by arc — 00–06 toy clock
  and SPD evaluation, 12–18 rings and writers, 19–24 computed answers / SAE /
  gradients / robustness / emotion, 25–31 months, numbers, and sensitivity
  batteries.
- **Cloud kernels (`kernels/`):** self-contained Kaggle script kernels for the
  7B runs.

## Environment

```bash
uv venv --python 3.13 && source .venv/bin/activate
uv pip install ripser persim scikit-learn scipy
# the toy-clock phase additionally uses goodfire-ai's `param-decomp` package
```

Local experiments run on MPS/CPU (models to 1.5B); 7B experiments run as
self-contained Kaggle script kernels (see `kernels/`). All randomness is
seeded.

## License

MIT (see [LICENSE](LICENSE)).
