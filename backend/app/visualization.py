"""Web ve masaüstü arayüzlerinin ortak görsel tanımları (renkler ve kesit katmanları).

Yalnız görüntüleme içindir; hiçbir maske veya metrik burada değişmez.
"""
from __future__ import annotations

import numpy as np

# Tümör alt bölgeleri (RGBA): 0 arka plan (saydam), 1 NCR yeşil, 2 ED sarı, 3 ET kırmızı.
TUMOR_COLORS = np.array([
    [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.65],
    [1.0, 1.0, 0.0, 0.65], [1.0, 0.0, 0.0, 0.75],
], dtype=np.float32)
# Konsensüs: tüm modeller (N/N) camgöbeği, çoğunluk mor, azınlık fuşya.
CONSENSUS_COLORS = {"all": (0.0, 0.82, 1.0, 0.80), "majority": (0.62, 0.42, 1.0, 0.80),
                    "minority": (1.0, 0.12, 0.62, 0.86)}
PROBABILITY_THRESHOLD = 0.15   # olasılık haritasında bu değerin altı gösterilmez
PROBABILITY_ALPHA = 0.55


def display_slice(raw: np.ndarray) -> np.ndarray:
    """Kesit başına 1–99 yüzdelik kontrast germe (yalnız beyin içi değerlerden)."""
    nonzero = raw[raw > 0]
    if not len(nonzero):
        return raw
    low, high = np.percentile(nonzero, [1, 99])
    return np.clip((np.clip(raw, low, high) - low) / (high - low + 1e-8), 0, 1)


def colorize(mask: np.ndarray) -> np.ndarray:
    """Etiket maskesini RGBA katmanına çevirir."""
    result = np.zeros((*mask.shape, 4), dtype=np.float32)
    for label in (1, 2, 3):
        result[mask == label] = TUMOR_COLORS[label]
    return result


def consensus_rgba(counts: np.ndarray, n_models: int) -> np.ndarray:
    """Voksel başına tümör bulan model sayısından RGBA konsensüs katmanı."""
    majority = counts > n_models / 2
    overlay = np.zeros((*counts.shape, 4), dtype=np.float32)
    overlay[counts == n_models] = CONSENSUS_COLORS["all"]
    overlay[majority & (counts < n_models)] = CONSENSUS_COLORS["majority"]
    overlay[(counts > 0) & ~majority] = CONSENSUS_COLORS["minority"]
    return overlay


def consensus_legend(n_models: int) -> str:
    if n_models > 2:
        return f"camgöbeği: {n_models}/{n_models} model · mor: çoğunluk · fuşya: tek model"
    return f"camgöbeği: {n_models}/{n_models} model · fuşya: tek model"
