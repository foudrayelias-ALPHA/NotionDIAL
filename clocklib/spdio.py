"""Interface to param_decomp: sink, data, config builder, decomposition runner + artifact IO."""

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, IterableDataset

from param_decomp.ci_fns import LayerwiseCiConfig
from param_decomp.configs import Cadence, OptimizerConfig, PDConfig, RuntimeConfig
from param_decomp.decomposition_targets import DecompositionTargetConfig
from param_decomp.metrics.faithfulness import FaithfulnessLossConfig
from param_decomp.metrics.importance_minimality import ImportanceMinimalityLossConfig
from param_decomp.metrics.stochastic_recon import StochasticReconLossConfig
from param_decomp.metrics.stochastic_recon_layerwise import StochasticReconLayerwiseLossConfig
from param_decomp.optimize import Trainer
from param_decomp.schedule import ScheduleConfig
from param_decomp.training_state import TrainingState

MODULES = ["W_e", "W_in", "W_out"]


class TwoHotPairDataset(IterableDataset[Tensor]):
    """Infinite uniform (a, b) pairs as two-hot vectors, shape (batch, p)."""

    def __init__(self, p: int, batch_size: int, seed: int):
        self.p, self.batch_size, self.seed = p, batch_size, seed

    def __iter__(self):
        from clocklib.model import two_hot

        g = torch.Generator().manual_seed(self.seed)
        while True:
            tokens = torch.randint(0, self.p, (self.batch_size, 2), generator=g)
            yield two_hot(tokens, self.p)


class LocalSink:
    """Minimal RunSink: jsonl metrics + stdout + final checkpoint only."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self._metrics = open(out_dir / "metrics.jsonl", "a")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        scalars = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        self._metrics.write(json.dumps({"step": step, **scalars}) + "\n")
        self._metrics.flush()

    def console(self, *lines: str) -> None:
        for line in lines:
            print(line)

    def checkpoint(self, snapshot: TrainingState) -> None:
        torch.save(snapshot.component_model, self.out_dir / f"model_{snapshot.step}.pth")

    def finish(self) -> None:
        self._metrics.close()


def run_tokens(model: nn.Module, batch: Tensor) -> Tensor:
    return model(batch)


def recon_mse(pred: Tensor, target: Tensor) -> tuple[Tensor, int]:
    d = (pred - target) ** 2
    return d.sum(), d.numel()


def build_pd_config(module_to_c: dict[str, int], steps: int, batch_size: int, seed: int,
                    imin_coeff: float = 1e-5, faith_coeff: float = 1.0,
                    warmup_steps: int = 1000, lr: float = 2e-3,
                    grad_clip: float | None = None, stoch_coeff: float = 1.0,
                    warmup_lr: float = 0.01) -> PDConfig:
    """resid_mlp1 recipe adapted to the modular adder (see decisions.md).

    Deviation from the lab recipe: an explicit FaithfulnessLoss in the main loop.
    The lab relies on warmup + the delta component; our edits are defined relative to
    the components, so a small delta is load-bearing here.
    """
    const_lr = ScheduleConfig(start_val=lr, fn_type="constant", warmup_pct=0.0)
    return PDConfig(
        seed=seed,
        n_mask_samples=1,
        ci_config=LayerwiseCiConfig(mode="layerwise", fn_type="mlp", hidden_dims=[16]),
        sigmoid_type="leaky_hard",
        decomposition_targets=[
            DecompositionTargetConfig(module_pattern=m, C=c) for m, c in module_to_c.items()
        ],
        identity_decomposition_targets=None,
        use_delta_component=True,
        tied_weights=None,
        loss_metrics=[
            ImportanceMinimalityLossConfig(coeff=imin_coeff, pnorm=2.0, beta=0.0),
            *([] if stoch_coeff == 0.0 else [
                StochasticReconLayerwiseLossConfig(coeff=stoch_coeff),
                StochasticReconLossConfig(coeff=stoch_coeff),
            ]),
            FaithfulnessLossConfig(coeff=faith_coeff),
        ],
        components_optimizer=OptimizerConfig(lr_schedule=const_lr, grad_clip_norm=grad_clip),
        ci_fn_optimizer=OptimizerConfig(lr_schedule=const_lr, grad_clip_norm=grad_clip),
        steps=steps,
        batch_size=batch_size,
        faithfulness_warmup_steps=warmup_steps,
        faithfulness_warmup_lr=warmup_lr,
        faithfulness_warmup_weight_decay=0.1,
    )


def decompose(target_model: nn.Module, p: int, module_to_c: dict[str, int], steps: int,
              batch_size: int, seed: int, out_dir: Path, device: str = "cpu",
              loader: DataLoader | None = None,
              ci_input_groups: dict[str, Tensor] | None = None,
              imin_coeff: float = 1e-5) -> dict:
    """Run SPD, then extract {module: U, V, W_target, delta, mean_ci} to decomposition.pt.

    `loader` defaults to the single-op two-hot stream. `ci_input_groups` optionally maps
    group names to input tensors; per-group mean CI is stored under mean_ci_<name>.
    """
    target_model.eval().requires_grad_(False)
    target_model.to(device)
    pd_config = build_pd_config(module_to_c, steps, batch_size, seed, imin_coeff=imin_coeff)
    runtime = RuntimeConfig(autocast_bf16=False, device=device, dp=None)
    trainer = Trainer(target_model=target_model, run_batch=run_tokens,
                      reconstruction_loss=recon_mse, pd_config=pd_config, runtime_config=runtime)
    if loader is None:
        loader = DataLoader(TwoHotPairDataset(p, batch_size, seed), batch_size=None)
    sink = LocalSink(out_dir)
    try:
        trainer.run(loader, sink, Cadence(train_log_every=500), eval_loop=None)
    finally:
        sink.finish()

    cm = trainer.component_model
    deltas = cm.calc_weight_deltas()
    art: dict[str, Any] = {"pd_config": pd_config.model_dump(), "p": p}
    for m in module_to_c:
        comp = cm.components[m]
        art[m] = {
            "U": comp.U.detach().cpu().clone(),      # (C, d_out)
            "V": comp.V.detach().cpu().clone(),      # (d_in, C)
            "W_target": cm.target_weight(m).detach().cpu().clone(),
            "delta": deltas[m].detach().cpu().clone(),
        }
    if ci_input_groups is None:
        from clocklib.model import all_pairs, two_hot

        tokens, _ = all_pairs(p)
        ci_input_groups = {"": two_hot(tokens, p)}
    for name, inputs in ci_input_groups.items():
        key = "mean_ci" if name == "" else f"mean_ci_{name}"
        art[key] = mean_causal_importance(cm, inputs)
    cm.cpu()
    torch.save(art, out_dir / "decomposition.pt")
    return art


@torch.no_grad()
def mean_causal_importance(cm, inputs: Tensor, chunk: int = 4096) -> dict[str, Tensor]:
    """Mean upper-leaky CI per component over the given inputs."""
    device = next(cm.parameters()).device
    inputs = inputs.to(device)
    sums: dict[str, Tensor] = {}
    counts: dict[str, int] = {}
    for i in range(0, inputs.shape[0], chunk):
        out = cm(inputs[i : i + chunk], cache_type="input")
        ci = cm.calc_causal_importances(out.cache, sampling="continuous")
        for m, v in ci.upper_leaky.items():
            flat = v.reshape(-1, v.shape[-1]).cpu()
            sums[m] = sums.get(m, torch.zeros(v.shape[-1])) + flat.sum(0)
            counts[m] = counts.get(m, 0) + flat.shape[0]
    return {m: sums[m] / counts[m] for m in sums}


def recon_weights(art: dict, module: str, include_delta: bool = False) -> Tensor:
    """Sum-of-rank-one reconstruction in the target weight's shape."""
    a = art[module]
    W = torch.einsum("ic,co->io", a["V"], a["U"])
    W = W if W.shape == a["W_target"].shape else W.T
    assert W.shape == a["W_target"].shape
    return W + a["delta"] if include_delta else W
