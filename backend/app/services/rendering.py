"""Kesit panelleri (PNG) ve özet rapor (PNG/PDF). Yalnız görüntüleme: hiçbir maske değişmez.

Uç noktalar iş parçacığı havuzunda eşzamanlı çalışır (bir ekran 8 paneli aynı anda ister):
- Rapor pyplot'un küresel durumu yerine nesne tabanlı ``Figure`` ile çizilir (pyplot iş parçacığı güvenli değildir).
- Oturum başına panel önbellekleri bir kilitle korunur; eşzamanlı ekleme/çıkarma ``KeyError`` üretiyordu.
"""
from __future__ import annotations

import io
import threading
from collections import OrderedDict

import matplotlib
import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from .. import pipeline
from .. import visualization as vis
from . import session


def _cache_lock(record: dict) -> threading.Lock:
    return record.setdefault("cache_lock", threading.Lock())   # dict.setdefault atomiktir


def make_panel(record: dict, source: str, slice_idx: int, modality: int) -> bytes:
    """Bir MR kesitini figür oluşturmadan, doğrudan hızlı PNG'ye dönüştürür."""
    lock = _cache_lock(record)
    cache: OrderedDict = record.setdefault("panel_cache", OrderedDict())
    base_cache: OrderedDict = record.setdefault("base_slice_cache", OrderedDict())
    key, base_key = (source, slice_idx, modality), (slice_idx, modality)
    with lock:
        cached = cache.get(key)
        if cached is not None:
            cache.move_to_end(key)
            return cached
        base = base_cache.get(base_key)
        if base is not None:
            base_cache.move_to_end(base_key)
    if base is None:
        raw = vis.display_slice(record["raw"][modality, :, :, slice_idx])
        base = np.repeat((raw * 255).round().astype(np.uint8)[..., None], 3, axis=2)
        with lock:
            base_cache[base_key] = base
            if len(base_cache) > 60:
                base_cache.popitem(last=False)
    pixels = base.astype(np.float32, copy=True)
    if source in record.get("mask_sources", ()):
        mask = record.get(source)
        if mask is not None:
            overlay = vis.colorize(mask[:, :, slice_idx])
            alpha = overlay[..., 3:4]
            pixels = pixels * (1.0 - alpha) + overlay[..., :3] * 255.0 * alpha
    elif source == "consensus":
        # Araştırma görünümü: kaç modelin tümör bulduğu; hiçbir maskeyi değiştirmez.
        overlay = vis.consensus_rgba(record["consensus"][:, :, slice_idx], record["n_models"])
        alpha = overlay[..., 3:4]
        pixels = pixels * (1.0 - alpha) + overlay[..., :3] * 255.0 * alpha
    elif source == "activation":
        probability = record.get("activation")
        if probability is not None:
            values = np.clip(probability[:, :, slice_idx], 0.0, 1.0)
            overlay = matplotlib.colormaps["jet"](values)[..., :3].astype(np.float32) * 255.0
            alpha = (values >= vis.PROBABILITY_THRESHOLD)[..., None].astype(np.float32) * vis.PROBABILITY_ALPHA
            pixels = pixels * (1.0 - alpha) + overlay * alpha
    out = io.BytesIO()
    # Matplotlib'deki origin="lower" görünümünü korumak için dikey eksen çevrilir.
    Image.fromarray(np.flipud(np.clip(pixels, 0, 255).astype(np.uint8)), mode="RGB").save(out, format="PNG", compress_level=1)
    payload = out.getvalue()
    with lock:
        cache[key] = payload
        if len(cache) > 180:
            cache.popitem(last=False)
    return payload


def render_report(record: dict, index: int, modality: int, fmt: str) -> bytes:
    """Tüm panelleri ve izlenebilirlik satırlarını (checkpoint SHA, çalışma yolu) içeren özet rapor."""
    engine, model_set = session.ENGINE, session.ENGINE.model_set
    names = {"original": "Orijinal MR", "gt": "Ground Truth", "ensemble": "Ensemble (ana maske)",
             "consensus": "Model Konsensüsü", "activation": "Tümör Olasılığı",
             **{key: f"{engine.name(key)} Tahmini" for key in engine.keys}}
    # Ekrandaki panel sırasıyla aynı: ana maske, kaynak MR, konsensüs, referans, tekil modeller iç test Dice sırasıyla.
    order = ["ensemble", "original", "consensus", "gt", "swin", "mednext", "unet"]
    sources = sorted(["original", *record["mask_sources"], "consensus"], key=lambda s: order.index(s) if s in order else len(order))
    fig = Figure(figsize=(4 * len(sources), 5.4), facecolor="#111827")
    axes = fig.subplots(1, len(sources))
    raw = vis.display_slice(record["raw"][modality, :, :, index])
    for number, (axis, source) in enumerate(zip(axes, sources), 1):
        axis.set_facecolor("#111827")
        axis.set_title(f"{number}. {names[source]}", color="#f8fafc", fontsize=10, fontweight="bold")
        if source == "consensus":
            axis.imshow(np.asarray(Image.open(io.BytesIO(make_panel(record, "consensus", index, modality))))[::-1], origin="lower")
        else:
            axis.imshow(raw, cmap="gray", origin="lower")
            if source != "original" and record.get(source) is not None:
                axis.imshow(vis.colorize(record[source][:, :, index]), origin="lower")
        axis.axis("off")
    fig.tight_layout(pad=1.5, rect=(0, 0.13, 1, 1))
    runtime = record.get("runtime", {})
    prints = session.model_fingerprints()
    modality_name = ("T1", "T1ce", "T2", "FLAIR")[modality]
    lines = [
        f"Vaka: {record.get('patient_name') or 'adsız'}  ·  Kesit: {index + 1}/{record['raw'].shape[3]} ({modality_name})  ·  Analiz: {runtime.get('created', '-')}",
        f"Model seti {model_set.version}: " + "  ·  ".join(f"{engine.name(k)} [{v['file']}, SHA-256 {v['sha256'][:12]}]" for k, v in prints.items()),
        *([f"Ensemble (ana maske): {pipeline.ENSEMBLE_METHOD_NAMES[model_set.ensemble['ensemble_method']]} · küçük ET eşiği "
           f"{model_set.ensemble['et_postprocessing'] or 'yok'} · validation'da kilitlendi · web hız modu: TTA yok · "
           f"konsensüs: {vis.consensus_legend(record['n_models'])}"] if "ensemble" in record else []),
        f"Çalışma: {runtime.get('device', '-')} · overlap {runtime.get('overlap', '-')} · yollar "
        + ", ".join(f"{k}={v}" for k, v in runtime.get("runtimes", {}).items())
        + f" · dosya yönelimi {runtime.get('orientation', '-')} · ön işleme: RAS + nonzero Z-score + ön plan kırpma (eğitim hattı)",
        "ARAŞTIRMA AMAÇLIDIR — klinik tanı veya tedavi kararı için tek başına kullanılamaz; uzman radyolog/hekim değerlendirmesi gerekir. "
        "Olasılık paneli tümör olasılık haritasıdır (Grad-CAM değildir).",
    ]
    for warning in runtime.get("warnings", []):
        lines.append(f"UYARI: {warning}")
    fig.text(0.01, 0.01, "\n".join(lines), color="#cbd5e1", fontsize=8.5, family="monospace", va="bottom")
    out = io.BytesIO()
    fig.savefig(out, format=fmt, dpi=200, facecolor=fig.get_facecolor())
    return out.getvalue()
