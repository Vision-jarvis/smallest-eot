"""Smart Turn v3.2 as a trainable PyTorch module, initialised from the published fp32 ONNX.

pipecat-ai publishes Smart Turn v3.2 only as ONNX (no PyTorch checkpoint), and its trainer
starts from openai/whisper-tiny. To *fine-tune* v3.2 instead of re-training from Whisper we
rebuild the architecture and copy the ONNX initialisers into it:

* Architecture: `SmartTurnV3` below is the same module graph as `SmartTurnV3Model` in
  github.com/pipecat-ai/smart-turn train.py (BSD-2-Clause, Copyright (c) 2024-2025 Daily):
  Whisper-tiny encoder (4 layers, d=384, 400 positions = 8 s), attention pooling, MLP head.
* Weights: named initialisers ("inner.encoder.conv1.weight", ...) map 1:1. The exporter
  anonymised the projection matrices ("val_17", ...) and stored them transposed, so we
  recover each one from the graph: a MatMul whose output feeds an Add with a named bias
  (q/v/out/fc1/fc2/pool) is that layer, and the bias-free MatMul between q and v is k_proj.
* Proof: `parity()` runs the ONNX and the torch module on the same features; the unit test
  (tests/test_smart_turn_torch.py) requires max |dp| < 1e-4.

The forward returns a logit (training uses BCE-with-logits); `prob()` matches the ONNX output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
ONNX_FP32 = ROOT / "data/hf/pipecat-ai__smart-turn-v3/smart-turn-v3.2-gpu.onnx"
N_MELS, N_FRAMES = 80, 800  # 8 s of 16 kHz audio, Whisper log-mel, 10 ms hop


def whisper_tiny_config():
    from transformers import WhisperConfig

    return WhisperConfig(
        d_model=384, encoder_layers=4, encoder_attention_heads=6, encoder_ffn_dim=1536,
        num_mel_bins=N_MELS, max_source_positions=400, dropout=0.0, attention_dropout=0.0,
        activation_function="gelu", attn_implementation="eager",
    )


class SmartTurnV3(nn.Module):
    """Whisper-tiny encoder + attention pooling + MLP; outputs one end-of-turn logit."""

    def __init__(self) -> None:
        super().__init__()
        from transformers.models.whisper.modeling_whisper import WhisperEncoder

        self.encoder = WhisperEncoder(whisper_tiny_config())
        h = 384
        self.pool_attention = nn.Sequential(nn.Linear(h, 256), nn.Tanh(), nn.Linear(256, 1))
        self.classifier = nn.Sequential(
            nn.Linear(h, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1),
        )

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        hs = self.encoder(input_features=input_features).last_hidden_state
        w = torch.softmax(self.pool_attention(hs), dim=1)
        return self.classifier((hs * w).sum(dim=1)).squeeze(-1)

    @torch.no_grad()
    def prob(self, input_features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(input_features))


def _onnx_state_dict(path: Path) -> dict[str, torch.Tensor]:
    import onnx
    from onnx import numpy_helper

    g = onnx.load(str(path)).graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    consumers: dict[str, list] = {}
    for n in g.node:
        for x in n.input:
            consumers.setdefault(x, []).append(n)
    sd = {k.removeprefix("inner."): torch.from_numpy(v.copy()) for k, v in init.items() if k.startswith("inner.")}
    pending_k: str | None = None
    for n in g.node:
        if n.op_type != "MatMul":
            continue
        w = next((x for x in n.input if x.startswith("val_") and x in init and init[x].ndim == 2), None)
        if w is None:
            continue
        bias = next((x for c in consumers.get(n.output[0], []) for x in c.input if x.startswith("inner.") and x.endswith(".bias")), None)
        if bias is None:  # k_proj (Whisper has no k bias): sits between q and v of the same layer
            if pending_k is None:
                raise ValueError(f"unexpected bias-free MatMul {w}")
            target, pending_k = pending_k, None
        else:
            target = bias.removeprefix("inner.").removesuffix(".bias") + ".weight"
            if target.endswith("q_proj.weight"):
                pending_k = target.replace("q_proj", "k_proj")
        sd[target] = torch.from_numpy(init[w].T.copy())
    return sd


def load_from_onnx(path: Path = ONNX_FP32) -> SmartTurnV3:
    """A SmartTurnV3 carrying exactly the published v3.2 fp32 weights."""
    m = SmartTurnV3()
    sd = _onnx_state_dict(path)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    # embed_positions is a fixed (non-trained) sinusoid in some transformers versions
    missing = [k for k in missing if not k.endswith("embed_positions.weight")]
    if missing or unexpected:
        raise ValueError(f"weight mapping incomplete: missing={missing} unexpected={unexpected}")
    return m.eval()


def parity(model: SmartTurnV3, path: Path = ONNX_FP32, n: int = 4, seed: int = 0) -> float:
    """max |p_torch - p_onnx| on log-mel-like random inputs (values in Whisper's [-1, 1.5] range)."""
    import onnxruntime as ort

    x = np.random.default_rng(seed).uniform(-1.0, 1.5, (n, N_MELS, N_FRAMES)).astype(np.float32)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    p_onnx = sess.run(None, {"input_features": x})[0].reshape(-1)
    p_t = model.eval().prob(torch.from_numpy(x)).numpy().reshape(-1)
    return float(np.abs(p_onnx - p_t).max())


def export_onnx(model: SmartTurnV3, out: Path, int8: bool = True) -> dict[str, Path]:
    """fp32 ONNX with Smart Turn's I/O contract (input_features [B,80,800] -> probs [B,1]),
    plus a dynamic int8 copy for CPU. Returns the written paths."""

    class _Wrap(nn.Module):
        def __init__(self, inner: SmartTurnV3) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, input_features: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(self.inner(input_features)).reshape(-1, 1)

    out.parent.mkdir(parents=True, exist_ok=True)
    w = _Wrap(model.eval().cpu())
    torch.onnx.export(w, (torch.zeros(2, N_MELS, N_FRAMES),), str(out), input_names=["input_features"], output_names=["logits"],
                      dynamic_axes={"input_features": {0: "batch"}, "logits": {0: "batch"}}, opset_version=18, dynamo=False)
    paths = {"fp32": out}
    if int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        q = out.with_name(out.stem + "_int8.onnx")
        quantize_dynamic(str(out), str(q), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])
        paths["int8"] = q
    return paths
