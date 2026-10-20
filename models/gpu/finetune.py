"""Fine-tune Smart Turn v3.2 on real Hindi pauses (GPU), with a CPU dry run that proves it trains.

    python -m models.gpu.finetune --dry-run                         # CPU, about 3 min, fake data allowed
    python -m models.gpu.finetune --config models/gpu/finetune_smart_turn_hi.yaml --data-root /data

What it does, in order:
1. Initialise `SmartTurnV3` with the published v3.2 fp32 weights (models/gpu/smart_turn_torch.py,
   parity with the ONNX is asserted before training starts).
2. Score the *untouched* v3.2 on the dev pauses (epoch 0). That is the number the fine-tune
   has to beat on real Hindi, measured with the same code.
3. Train with BCE-with-logits (positive weight from the training-set class ratio), AdamW,
   two learning rates (encoder / pooling+head), linear warm-up then cosine, encoder frozen for
   the first `freeze_encoder_epochs`, mu-law 8 kHz and MUSAN augmentation on the waveform,
   bf16 autocast on CUDA.
4. After every epoch, dev Lat@5% / FCR@300 / AUC on IndicVoices dev pauses through
   eval/metrics.py (eot-bench definitions, score point 0.2 s), plus clip AUC on the Smart Turn
   test-split Hindi clips. Keep the best dev Lat@5%; stop after `patience` bad epochs.
5. Export the best weights to fp32 ONNX (same I/O as Smart Turn: input_features [B,80,800] ->
   probability [B,1]) and dynamic int8, check torch vs ONNX parity, and write run_summary.json
   (per-epoch metrics, data counts, throughput, wall time and the dollar cost of the run).

The IndicVoices test split and both halves of eot-bench hi are never read here.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from core.audio import decode, mix_at_snr, mulaw_roundtrip, resample
from core.encoders import log_mel, right_align
from eval import metrics
from models.gpu.smart_turn_torch import SmartTurnV3, export_onnx, load_from_onnx, parity

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).with_name("finetune_smart_turn_hi.yaml")
SR = 16000


def read_sources(entries: list[dict], root: Path, limit: int | None = None) -> pd.DataFrame:
    """Concatenate Smart Turn-schema parquet sources (audio, endpoint_bool, ...)."""
    parts = []
    for e in entries:
        files = sorted(glob.glob(str(root / e["glob"])))
        if not files:
            print(f"  [skip] {e['name']}: no files for {e['glob']}")
            continue
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        if e.get("max_rows"):
            df = df.sample(n=min(len(df), e["max_rows"]), random_state=0)
        df = pd.concat([df] * int(e.get("repeat", 1)), ignore_index=True).assign(source=e["name"])
        parts.append(df)
        print(f"  {e['name']}: {len(df)} clips, {df['endpoint_bool'].mean():.1%} end-of-turn")
    if not parts:
        raise SystemExit("no training data found (see STATUS.md: IndicVoices pauses are built by bharat_turn.indicvoices_episodes)")
    df = pd.concat(parts, ignore_index=True)
    return df.sample(n=min(limit, len(df)), random_state=0).reset_index(drop=True) if limit else df


@dataclass
class Augment:
    mulaw8k_p: float = 0.0
    musan_p: float = 0.0
    snr_db: tuple[float, float] = (5.0, 20.0)
    noise_dir: Path | None = None

    def __post_init__(self) -> None:
        self.noise = sorted(self.noise_dir.rglob("*.wav")) if self.noise_dir and self.noise_dir.exists() else []
        if self.musan_p and not self.noise:
            print("  [warn] MUSAN noise not found: noise augmentation disabled")

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.noise and rng.random() < self.musan_p:
            import soundfile as sf

            n, s = sf.read(self.noise[int(rng.integers(len(self.noise)))], dtype="float32")
            n = resample(n.mean(axis=1) if n.ndim > 1 else n, s, SR)
            x = mix_at_snr(x, n, float(rng.uniform(*self.snr_db)), rng)
        if rng.random() < self.mulaw8k_p:
            x = mulaw_roundtrip(x, SR)
        return x


class ClipDataset(Dataset):
    """Smart Turn-schema rows -> (log-mel [80, 800], label). Waveform augmentation per epoch."""

    def __init__(self, df: pd.DataFrame, aug: Augment | None = None, seed: int = 0) -> None:
        self.audio = df["audio"].tolist()
        self.y = df["endpoint_bool"].astype(np.float32).to_numpy()
        self.aug, self.seed, self.epoch = aug, seed, 0

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        x, sr = decode(self.audio[i])
        x = right_align(resample(x, sr, SR))  # last 8 s, left zero-padded (Smart Turn convention)
        if self.aug is not None:
            x = self.aug(x, np.random.default_rng((self.seed, self.epoch, i)))
        return torch.from_numpy(log_mel(x, normalise=True).astype(np.float32)), torch.tensor(self.y[i])


def predict(model: SmartTurnV3, ds: ClipDataset, device: str, bs: int, workers: int) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for x, _ in DataLoader(ds, batch_size=bs, num_workers=workers):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                out.append(torch.sigmoid(model(x.to(device))).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def dev_pause_metrics(df: pd.DataFrame, p: np.ndarray) -> dict:
    """eot-bench metrics on pause windows (score point 0.2 s; one window per span)."""
    rows = []
    for r, pi in zip(df.itertuples(index=False), p):
        base = {"id": r.rec_id, "span_index": int(r.span_index), "label": r.label, "timestamp": 0.0}
        rows.append({**base, "silence_dur": float(r.offset_s), "p_eot": float(pi)})
        rows.append({**base, "silence_dur": float(r.span_dur), "p_eot": 0.0})  # span length only
    pred = pd.DataFrame(rows)
    _, s = metrics.summarise(pred, score_point=0.2)
    return {k: s[k] for k in ("Lat@5%", "FCR@300", "FCR@600", "auc", "ap", "n_hold", "n_eot")}


def clip_auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else math.nan


def lr_lambda(total: int, warmup: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))

    return f


def evaluate(model, dev_pauses, dev_pause_ds, dev_clip_ds, device, bs, workers) -> dict:
    out: dict = {}
    if dev_pause_ds is not None:
        out["dev_pauses"] = dev_pause_metrics(dev_pauses, predict(model, dev_pause_ds, device, bs, workers))
    if dev_clip_ds is not None:
        out["dev_clips_auc"] = clip_auc(dev_clip_ds.y, predict(model, dev_clip_ds, device, bs, workers))
    return out


def score_of(ev: dict) -> tuple[float, float]:
    """Lower is better: dev Lat@5% (inf if the 5% budget is unreachable), then -AUC."""
    dp = ev.get("dev_pauses") or {}
    lat = dp.get("Lat@5%")
    auc = dp.get("auc") if dp else ev.get("dev_clips_auc")
    return (math.inf if lat is None else float(lat), -(auc if auc is not None and not math.isnan(auc) else 0.0))


def overfit_check(model: SmartTurnV3, ds: ClipDataset, device: str, steps: int = 30) -> dict:
    """Gradient sanity check: full-model training on 16 clips must drive their loss down."""
    import copy

    m = copy.deepcopy(model).to(device).train()
    x, y = next(iter(DataLoader(ds, batch_size=len(ds))))
    x, y = x.to(device), y.to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    curve = []
    for _ in range(steps):
        loss = nn.functional.binary_cross_entropy_with_logits(m(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        curve.append(loss.item())
    ok = curve[-1] < 0.5 * curve[0]
    print(f"  overfit check on {len(ds)} clips: loss {curve[0]:.4f} -> {curve[-1]:.4f} ({'ok' if ok else 'FAILED'})")
    return {"n_clips": len(ds), "steps": steps, "loss_first": curve[0], "loss_last": curve[-1], "passed": ok}


def run(cfg: dict, data_root: Path, out_dir: Path, device: str, dry_run: bool) -> dict:
    t_start = time.time()
    tr, ag = cfg["train"], cfg["augment"]
    torch.manual_seed(tr["seed"])
    limit = 50 if dry_run else None
    print("train sources:")
    train_df = read_sources(cfg["data"]["train"], data_root, limit)
    dev_pauses = None
    dp = data_root / cfg["data"]["dev_pauses"]
    if dp.exists():
        dev_pauses = pd.read_parquet(dp)
        dev_pauses = dev_pauses[np.isclose(dev_pauses["offset_s"], 0.2)].reset_index(drop=True)
        if dry_run:
            dev_pauses = dev_pauses.head(40)
    else:
        print(f"  [warn] no dev pauses at {dp}: early stopping falls back to clip AUC")
    dev_clips = read_sources([{"name": "dev_clips", "glob": cfg["data"]["dev_clips"]}], data_root, 40 if dry_run else None)
    aug = Augment(ag["mulaw8k_p"], ag["musan_p"], tuple(ag["snr_db"]), data_root / "data/musan/musan/noise")
    train_ds = ClipDataset(train_df, aug, tr["seed"])
    workers = 0 if dry_run else tr["num_workers"]
    bs = 8 if dry_run else tr["batch_size"]
    dev_pause_ds = ClipDataset(dev_pauses) if dev_pauses is not None and len(dev_pauses) else None
    dev_clip_ds = ClipDataset(dev_clips)

    model = load_from_onnx(data_root / cfg["init"]["onnx_fp32"])
    dp0 = parity(model, data_root / cfg["init"]["onnx_fp32"])
    assert dp0 < 1e-4, f"torch/ONNX parity failed before training: {dp0}"
    model.to(device)
    pos = float(train_ds.y.mean())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / max(pos, 1e-6), device=device))
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": tr["lr_encoder"]}, {"params": head, "lr": tr["lr_head"]}], weight_decay=tr["weight_decay"])
    epochs = 1 if dry_run else tr["epochs"]
    loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=workers, drop_last=len(train_ds) > bs,
                        pin_memory=device == "cuda", persistent_workers=workers > 0)
    total = epochs * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(total, int(tr["warmup_frac"] * total)))

    history = [{"epoch": 0, "note": "published Smart Turn v3.2, before fine-tuning", **evaluate(model, dev_pauses, dev_pause_ds, dev_clip_ds, device, bs, workers)}]
    print(json.dumps(history[-1], default=float))
    best, best_state, bad, step, losses = score_of(history[0]), None, 0, 0, []
    t_train = time.time()
    for ep in range(1, epochs + 1):
        train_ds.epoch = ep
        frozen = ep <= cfg["init"]["freeze_encoder_epochs"] and not dry_run
        for p in enc:
            p.requires_grad_(not frozen)
        model.train()
        ep_loss = []
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logit = model(x)
            loss = loss_fn(logit.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
            opt.step()
            sched.step()
            step += 1
            ep_loss.append(loss.item())
            if step % 50 == 0 or dry_run:
                print(f"  epoch {ep} step {step}/{total} loss {ep_loss[-1]:.4f}", flush=True)
        losses += ep_loss
        ev = {"epoch": ep, "encoder_frozen": frozen, "train_loss": float(np.mean(ep_loss)), **evaluate(model, dev_pauses, dev_pause_ds, dev_clip_ds, device, bs, workers)}
        history.append(ev)
        print(json.dumps(ev, default=float), flush=True)
        sc = score_of(ev)
        if sc < best:
            best, bad = sc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tr["patience"]:
                print("  early stop")
                break
    train_s = time.time() - t_train
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"dry_run": dry_run, "device": device, "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
                     "n_train": len(train_ds), "train_sources": train_df["source"].value_counts().to_dict(),
                     "n_dev_pauses": 0 if dev_pauses is None else len(dev_pauses), "n_dev_clips": len(dev_clip_ds),
                     "parity_before_training": dp0, "history": history, "steps": step, "train_seconds": train_s,
                     "samples_per_second": step * bs / max(train_s, 1e-9), "first_losses": losses[:5], "last_losses": losses[-5:]}
    if dry_run:
        src = train_df[train_df.source == cfg["data"]["train"][0]["name"]]  # real-Hindi pauses (or their fake stand-in)
        src = src if len(src) >= 16 else train_df
        pick = pd.concat([src[src.endpoint_bool].head(8), src[~src.endpoint_bool].head(8)])
        summary["overfit_check"] = overfit_check(model, ClipDataset(pick), device)
        if best_state is None:  # exercise the export path anyway; a dry run is never a result
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        summary["result"] = "no epoch beat the published v3.2 on dev; nothing exported"
    else:
        model.load_state_dict(best_state)
        model.cpu()
        torch.save(best_state, out_dir / "best.pt")
        paths = export_onnx(model, out_dir / "smart_turn_hi_v1.onnx", int8=cfg["export"]["int8"])
        summary["exports"] = {k: str(v) for k, v in paths.items()}
        summary["parity_torch_vs_onnx_fp32"] = parity(model, paths["fp32"])
        if "int8" in paths:
            summary["max_dp_int8_vs_torch"] = parity(model, paths["int8"])
        summary["result"] = f"best dev score {best} exported"
    c = cfg["cost"]
    wall_h = (time.time() - t_start) / 3600
    summary["wall_hours"] = wall_h
    summary["cost_usd_if_on_" + c["gpu"]] = wall_h * (c["usd_per_hour"] + c["cpu_cores"] * 0.04716 + c["mem_gib"] * 0.007992)
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: summary[k] for k in ("result", "steps", "train_seconds", "samples_per_second", "first_losses", "last_losses")}, default=float))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--data-root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true", help="CPU, 50 training clips, 1 epoch, batch 8")
    ap.add_argument("--fake-indicvoices", action="store_true", help="dry run on bharat_turn.fake_indicvoices pauses (pipeline test only)")
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    if a.fake_indicvoices:
        cfg["data"]["train"][0]["glob"] = "data/_fake_indicvoices/pauses/train.parquet"
        cfg["data"]["dev_pauses"] = "data/_fake_indicvoices/pauses/dev.parquet"
    device = "cuda" if torch.cuda.is_available() and not a.dry_run else "cpu"
    if not a.dry_run and device != "cuda":
        raise SystemExit("A full run needs a CUDA GPU: `modal run models/gpu/modal_app.py` (or --dry-run on CPU).")
    out = a.out or (ROOT / ("data/_dryrun_finetune" if a.dry_run else cfg["export"]["out_dir"]))
    run(cfg, a.data_root, out, device, a.dry_run)


if __name__ == "__main__":
    main()
