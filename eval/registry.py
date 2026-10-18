"""Name -> detector factory. Every system in the report is built here."""

from __future__ import annotations


def make(name: str):
    from adapters import baselines

    if name == "silero_timer":
        return baselines.SileroTimer()
    if name == "smart_turn_v32":
        return baselines.SmartTurnDetector()
    if name == "smart_turn_v32_fp32":
        return baselines.SmartTurnFp32Detector()
    if name == "hinglish_turn":
        return baselines.HinglishTurnDetector()
    if name.startswith("pulse_eot_hi"):
        from models.pulse_eot_hi import PulseEoTHi

        # e.g. "pulse_eot_hi" (default model) or "pulse_eot_hi:audio_only" (ablation)
        variant = name.split(":", 1)[1] if ":" in name else "full"
        return PulseEoTHi.load(variant)
    raise KeyError(name)
