"""Ortak görsel tanımlar (backend/app/visualization.py): renkler, konsensüs katmanı, kontrast germe."""
import numpy as np
import pytest

from backend.app import visualization as vis


def test_colorize_maps_each_label_and_keeps_background_transparent():
    mask = np.array([[0, 1], [2, 3]])
    rgba = vis.colorize(mask)
    assert rgba.shape == (2, 2, 4)
    assert rgba[0, 0, 3] == 0
    for (i, j), label in {(0, 1): 1, (1, 0): 2, (1, 1): 3}.items():
        assert np.array_equal(rgba[i, j], vis.TUMOR_COLORS[label])


def test_consensus_layer_separates_all_majority_and_minority():
    counts = np.array([0, 1, 2, 3])
    overlay = vis.consensus_rgba(counts, n_models=3)
    assert overlay[0, 3] == 0
    assert tuple(overlay[1]) == vis.CONSENSUS_COLORS["minority"]
    assert tuple(overlay[2]) == vis.CONSENSUS_COLORS["majority"]
    assert tuple(overlay[3]) == vis.CONSENSUS_COLORS["all"]
    # İki modelde "çoğunluk" katmanı yoktur: 1/2 azınlık, 2/2 tam uyum.
    two = vis.consensus_rgba(np.array([1, 2]), n_models=2)
    assert tuple(two[0]) == vis.CONSENSUS_COLORS["minority"] and tuple(two[1]) == vis.CONSENSUS_COLORS["all"]
    assert "mor" in vis.consensus_legend(3) and "mor" not in vis.consensus_legend(2)


def test_display_slice_stretches_to_unit_range_and_ignores_empty():
    rng = np.random.default_rng(0)
    raw = np.zeros((20, 20), dtype=np.float32)
    raw[5:15, 5:15] = rng.uniform(10, 500, size=(10, 10))
    out = vis.display_slice(raw)
    assert out.min() >= 0 and out.max() <= 1 and out.max() > 0.99
    empty = np.zeros((4, 4))
    assert vis.display_slice(empty) is empty


def test_panel_cache_is_safe_under_concurrent_requests():
    # Ekran 8 paneli aynı anda ister; kilitsiz önbellekte eşzamanlı ekleme/çıkarma KeyError veriyordu.
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    pytest.importorskip("backend.app.pipeline", reason="çekirdek public vitrinde yok (docs/PUBLIC_VITRIN_YOL_HARITASI.md §2)")
    from backend.app.services import rendering

    raw = np.random.default_rng(0).random((4, 16, 16, 40), dtype=np.float32)
    mask = (raw[3] > 0.7).astype(np.int32)
    record = {"raw": raw, "swin": mask, "mask_sources": ["swin"], "consensus": mask.astype(np.uint8), "n_models": 1,
              "activation": raw[0]}
    jobs = [(source, z, m) for z in range(40) for m in range(4) for source in ("original", "swin", "consensus", "activation")]
    with ThreadPoolExecutor(max_workers=8) as pool:
        payloads = list(pool.map(lambda job: rendering.make_panel(record, *job), jobs * 2))
    assert all(p[:8] == b"\x89PNG\r\n\x1a\n" for p in payloads)
    assert len(record["panel_cache"]) <= 180 and len(record["base_slice_cache"]) <= 60
