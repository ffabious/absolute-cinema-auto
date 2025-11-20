from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np


@dataclass
class BeatAnalysis:
    tempo: float
    beat_times: List[float]
    confidence: float
    duration: float

    def to_dict(self) -> Dict:
        return {
            "tempo": self.tempo,
            "beat_times": self.beat_times,
            "confidence": self.confidence,
            "duration": self.duration,
        }


class BeatAnalyzer:
    """Detect beat grid for soundtrack and expose timestamps for shot planning."""

    def __init__(self, music_path: Path) -> None:
        self.music_path = music_path

    def analyze(self) -> BeatAnalysis:
        y, sr = librosa.load(self.music_path, sr=None, mono=True)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, tightness=400, trim=False)
        tempo_val = float(tempo)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
        tempo_confidence = self._tempo_confidence(y, sr, tempo_val)
        duration = float(len(y) / sr)
        return BeatAnalysis(
            tempo=tempo_val,
            beat_times=beat_times,
            confidence=tempo_confidence,
            duration=duration,
        )

    def _tempo_confidence(self, signal: np.ndarray, sr: int, tempo: float) -> float:
        hop_length = 512
        onset_env = librosa.onset.onset_strength(y=signal, sr=sr, hop_length=hop_length)
        autocorr = np.correlate(onset_env, onset_env, mode="full")
        autocorr = autocorr[autocorr.size // 2 :]
        tempo_period = 60.0 / max(tempo, 1e-6)
        tempo_frames = int(round(tempo_period * sr / hop_length))
        peak = autocorr[tempo_frames] if tempo_frames < len(autocorr) else 0.0
        norm = autocorr.max() if autocorr.size else 1.0
        return float(peak / max(norm, 1e-6))
