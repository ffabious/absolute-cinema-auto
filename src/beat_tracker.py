from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np


@dataclass
class BeatAnalysis:
    tempo: float
    beat_times: List[float]
    beat_strengths: List[float]  # Relative strength of each beat (0-1)
    downbeat_indices: List[int]  # Indices of strong/downbeats
    confidence: float
    duration: float

    def to_dict(self) -> Dict:
        return {
            "tempo": self.tempo,
            "beat_times": self.beat_times,
            "beat_strengths": self.beat_strengths,
            "downbeat_indices": self.downbeat_indices,
            "confidence": self.confidence,
            "duration": self.duration,
        }


class BeatAnalyzer:
    """Detect beat grid for soundtrack and expose timestamps for shot planning.
    
    This analyzer identifies not just beat times but also:
    - Beat strength (how loud/prominent each beat is)
    - Downbeats (strong beats that start musical phrases, typically every 4 or 8 beats)
    """

    def __init__(self, music_path: Path) -> None:
        self.music_path = music_path

    def analyze(self) -> BeatAnalysis:
        y, sr = librosa.load(self.music_path, sr=None, mono=True)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, tightness=400, trim=False)
        tempo_val = float(tempo)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
        tempo_confidence = self._tempo_confidence(y, sr, tempo_val)
        duration = float(len(y) / sr)
        
        # Compute beat strengths and identify downbeats
        beat_strengths = self._compute_beat_strengths(y, sr, beat_frames)
        downbeat_indices = self._identify_downbeats(y, sr, beat_frames, beat_strengths)
        
        print(f"[BeatAnalyzer] Found {len(beat_times)} beats, {len(downbeat_indices)} downbeats")
        
        return BeatAnalysis(
            tempo=tempo_val,
            beat_times=beat_times,
            beat_strengths=beat_strengths,
            downbeat_indices=downbeat_indices,
            confidence=tempo_confidence,
            duration=duration,
        )

    def _compute_beat_strengths(self, y: np.ndarray, sr: int, beat_frames: np.ndarray) -> List[float]:
        """Compute the relative strength (energy) of each beat.
        
        Uses onset envelope strength at each beat position to determine
        how prominent/loud each beat is.
        """
        if len(beat_frames) == 0:
            return []
        
        hop_length = 512
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        
        # Get onset strength at each beat frame
        strengths = []
        for frame in beat_frames:
            if frame < len(onset_env):
                strengths.append(float(onset_env[frame]))
            else:
                strengths.append(0.0)
        
        # Normalize to 0-1 range
        if strengths:
            max_strength = max(strengths)
            min_strength = min(strengths)
            range_val = max_strength - min_strength
            if range_val > 0:
                strengths = [(s - min_strength) / range_val for s in strengths]
            else:
                strengths = [0.5] * len(strengths)
        
        return strengths
    
    def _identify_downbeats(self, y: np.ndarray, sr: int, beat_frames: np.ndarray, 
                            beat_strengths: List[float]) -> List[int]:
        """Identify downbeats (strong beats that start musical phrases).
        
        Uses multiple signals:
        1. Beat strength from onset envelope
        2. Spectral flux (sudden spectral changes)
        3. Regular patterns (every 4 or 8 beats in common time)
        4. RMS energy changes
        
        Downbeats are beats where:
        - Energy is significantly higher than surrounding beats, OR
        - There's a significant change in spectral content, OR
        - They fall on predicted phrase boundaries (every 4/8 beats) with high confidence
        """
        if len(beat_frames) < 4:
            return list(range(0, len(beat_frames), 4))  # Fallback: every 4th beat
        
        hop_length = 512
        
        # Compute RMS energy
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        
        # Compute spectral centroid for detecting timbral changes
        spectral = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        
        # Score each beat for "downbeat-ness"
        downbeat_scores = []
        
        for i, frame in enumerate(beat_frames):
            score = 0.0
            
            # 1. Beat strength contribution (0-0.4)
            if i < len(beat_strengths):
                score += beat_strengths[i] * 0.4
            
            # 2. Local energy peak (0-0.3)
            if frame < len(rms):
                window = 4
                start = max(0, i - window)
                end = min(len(beat_frames), i + window + 1)
                local_frames = [beat_frames[j] for j in range(start, end) if beat_frames[j] < len(rms)]
                if local_frames:
                    local_rms = [rms[f] for f in local_frames if f < len(rms)]
                    if local_rms and rms[frame] >= max(local_rms) * 0.95:
                        score += 0.3
            
            # 3. Spectral change (sudden brightness/timbre change) (0-0.2)
            if i > 0 and frame < len(spectral) and beat_frames[i-1] < len(spectral):
                prev_frame = beat_frames[i-1]
                spectral_diff = abs(spectral[frame] - spectral[prev_frame])
                # Normalize by typical spectral range
                spectral_range = spectral.max() - spectral.min() if spectral.max() > spectral.min() else 1.0
                score += min(0.2, (spectral_diff / spectral_range) * 0.4)
            
            # 4. Phrase boundary bonus (every 4 or 8 beats from start) (0-0.2)
            if i % 4 == 0:
                score += 0.1
            if i % 8 == 0:
                score += 0.1
            
            downbeat_scores.append(score)
        
        # Identify downbeats: beats with high scores
        # Use adaptive threshold based on score distribution
        if not downbeat_scores:
            return []
        
        mean_score = np.mean(downbeat_scores)
        std_score = np.std(downbeat_scores)
        threshold = mean_score + 0.3 * std_score  # Above average beats
        
        downbeat_indices = []
        min_gap = 3  # Minimum 3 beats between downbeats
        
        last_downbeat = -min_gap
        for i, score in enumerate(downbeat_scores):
            if score >= threshold and (i - last_downbeat) >= min_gap:
                downbeat_indices.append(i)
                last_downbeat = i
        
        # Ensure we have at least one downbeat every 8 beats
        if len(downbeat_indices) < len(beat_frames) // 8:
            # Fall back to phrase-based downbeats
            downbeat_indices = list(range(0, len(beat_frames), 4))
        
        # Always include the first beat as a downbeat
        if 0 not in downbeat_indices:
            downbeat_indices.insert(0, 0)
        
        return sorted(downbeat_indices)

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
