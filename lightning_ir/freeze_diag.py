"""Drop-in Lightning callback to diagnose the frozen ColBERT/MVR training loss on NeoBERT.

Logs, for the first `max_steps` training steps, exactly the four things that distinguish the
possible freeze mechanisms — so one short run tells you which it is:

  * scores    -> std / min / max / %finite   (std -> 0 == score collapse: all docs score alike,
                 MaxSim carries no gradient -> loss frozen)
  * loss      -> the value, and whether it is finite
  * gradients -> total / backbone / projection L2 norm, %finite, and how many params have an
                 (near-)zero grad   (many zeros in bf16 but not fp32 == gradient underflow, the
                 prime suspect for a bf16-mixed-only freeze that fp32 does not show)

Reading the output:
  - scores std healthy (rising) + loss descends  -> not frozen (something else / config)
  - scores std -> 0                               -> collapse; look at the head / normalization
  - NONFINITE on scores or grads                  -> numerical blowup (bf16); try precision:32
  - grads finite but backbone grad ~0 / many zero -> underflow / dead backbone; try precision:32
  - loss high & flat but scores well-spread       -> target/teacher-scale mismatch, not the model

Logs every metric to your configured logger (wandb) under the `freezediag/` prefix as a
per-step time-series (plot `freezediag/score_std` and `freezediag/grad_backbone`), and also
prints to stdout as a durable fallback in the SLURM log. Rank-0 only, costs ~nothing,
auto-silences after `max_steps`. Does NOT change training.

Ships inside the lightning_ir package, so a normal `pip install` makes it importable — no
run-dir copy, no PYTHONPATH. Reference it in your ColBERT config by its package path:

    trainer:
      callbacks:
        - class_path: lightning_ir.freeze_diag.FreezeDiagnostics
          init_args:
            max_steps: 300
"""

from __future__ import annotations

from typing import Any

import torch
from lightning.pytorch.callbacks import Callback


def _finite_frac(t: torch.Tensor) -> float:
    return float(torch.isfinite(t).float().mean()) if t.numel() else float("nan")


class FreezeDiagnostics(Callback):
    def __init__(self, max_steps: int = 300, every_n: int = 1, zero_grad_eps: float = 1e-12):
        """Args:
        max_steps: stop logging after this many optimizer steps (keeps the log short).
        every_n: log every n-th step (1 = every step).
        zero_grad_eps: a per-parameter grad L2 norm below this counts as an (underflowed) zero.
        """
        super().__init__()
        self.max_steps = max_steps
        self.every_n = max(1, every_n)
        self.zero_grad_eps = zero_grad_eps
        self._printed_header = False
        self._grad_stats: dict[str, float] = {}

    def _active(self, trainer) -> bool:
        return (
            trainer.is_global_zero
            and trainer.global_step <= self.max_steps
            and trainer.global_step % self.every_n == 0
        )

    def on_train_start(self, trainer, pl_module) -> None:
        if trainer.is_global_zero and not self._printed_header:
            self._printed_header = True
            print(
                "[freeze-diag] logging first "
                f"{self.max_steps} steps: scores(std/min/max/finite), loss, "
                "grads(total/backbone/proj, finite, #zero) — watch score-std -> 0, NONFINITE, "
                "or backbone-grad -> 0.",
                flush=True,
            )

    @torch.no_grad()
    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        # Grads are populated here (post-backward, pre-step). Stash stats; log them in
        # on_train_batch_end where logger context is guaranteed available.
        if not self._active(trainer):
            return
        tot = bb = proj = 0.0
        n_with_grad = n_zero = n_nonfinite = 0
        for name, p in pl_module.named_parameters():
            if p.grad is None:
                continue
            n_with_grad += 1
            g = p.grad.detach().float()
            if not torch.isfinite(g).all():
                n_nonfinite += 1
            gn2 = float(g.pow(2).sum())
            tot += gn2
            if "projection" in name:
                proj += gn2
            else:
                bb += gn2
            if gn2**0.5 < self.zero_grad_eps:
                n_zero += 1
        self._grad_stats = {
            "grad_total": tot**0.5,
            "grad_backbone": bb**0.5,
            "grad_proj": proj**0.5,
            "grad_nonfinite_params": float(n_nonfinite),
            "n_zero_grad": float(n_zero),
            "n_params_with_grad": float(n_with_grad),
        }

    @torch.no_grad()
    def on_train_batch_end(
        self, trainer, pl_module, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        if not self._active(trainer):
            return
        step = trainer.global_step
        loss = outputs.get("loss") if isinstance(outputs, dict) else None
        output = outputs.get("output") if isinstance(outputs, dict) else None
        scores = getattr(output, "scores", None)

        metrics: dict[str, float] = dict(self._grad_stats)
        self._grad_stats = {}

        if scores is None:
            print(f"[freeze-diag] step={step:5d}  (no scores in output)", flush=True)
        else:
            s = scores.detach().float().flatten()
            finite = _finite_frac(s)
            s_ok = s[torch.isfinite(s)]
            std = float(s_ok.std()) if s_ok.numel() > 1 else float("nan")
            smin = float(s_ok.min()) if s_ok.numel() else float("nan")
            smax = float(s_ok.max()) if s_ok.numel() else float("nan")
            lval = float(loss.detach()) if isinstance(loss, torch.Tensor) else float("nan")
            metrics.update(
                loss=lval, score_std=std, score_min=smin, score_max=smax, score_finite=finite
            )
            g = self._fmt_grad(metrics)
            flag = "  <-- NONFINITE" if finite < 1.0 else ("  <-- SCORE-STD~0" if std < 1e-3 else "")
            print(
                f"[freeze-diag] step={step:5d}  loss={lval:9.4f}  "
                f"scores n={s.numel():4d} std={std:8.4f} min={smin:8.3f} max={smax:8.3f} "
                f"finite={finite:5.1%}  {g}{flag}",
                flush=True,
            )

        # Route to the configured logger (wandb) as a per-step time-series.
        if metrics:
            pl_module.log_dict(
                {f"freezediag/{k}": v for k, v in metrics.items()},
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                rank_zero_only=True,
                batch_size=1,
            )

    @staticmethod
    def _fmt_grad(m: dict[str, float]) -> str:
        if "grad_total" not in m:
            return "grad=?"
        flag = ""
        if m.get("grad_nonfinite_params"):
            flag = f" <-- {int(m['grad_nonfinite_params'])} NONFINITE-GRAD"
        elif m.get("grad_backbone", 1.0) < 1e-8:
            flag = " <-- BACKBONE GRAD ~0"
        return (
            f"grad tot={m['grad_total']:.2e} bb={m['grad_backbone']:.2e} "
            f"proj={m['grad_proj']:.2e} #zero={int(m.get('n_zero_grad', 0))}{flag}"
        )
