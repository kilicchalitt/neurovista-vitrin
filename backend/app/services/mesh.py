"""3B görünüm için yüzey ağları (Marching Cubes). Yalnız görüntüleme: kaynak hacim ve maskeler değişmez."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, binary_opening, gaussian_filter, label
from skimage.measure import marching_cubes

from .. import inference_core as core

MIN_VOXELS = 5


def clean_brain_surface_mask(flair: np.ndarray) -> np.ndarray:
    """Ham FLAIR sıfır-maskesindeki küçük gürültüleri silip gerçek beyin hacmini korur."""
    mask = np.asarray(flair > 0, dtype=bool)
    mask = binary_opening(mask, iterations=1)
    mask = binary_closing(mask, iterations=2)
    mask = binary_fill_holes(mask)
    components, count = label(mask)
    if count:
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        mask = components == sizes.argmax()
    return mask


def laplacian_smooth(vertices: np.ndarray, faces: np.ndarray, iterations: int = 0, factor: float = .24) -> np.ndarray:
    """Yalnızca görüntüleme mesh'ini yumuşatır; orijinal MR/maske verisine dokunmaz."""
    if iterations <= 0:
        return vertices
    neighbours = [set() for _ in range(len(vertices))]
    for a, b, c in faces:
        neighbours[a].update((b, c))
        neighbours[b].update((a, c))
        neighbours[c].update((a, b))
    result = vertices.astype(np.float32, copy=True)
    for _ in range(iterations):
        updated = result.copy()
        for index, adjacent in enumerate(neighbours):
            if adjacent:
                updated[index] = result[index] * (1.0 - factor) + result[list(adjacent)].mean(axis=0) * factor
        result = updated
    return result


def mesh_payload(binary: np.ndarray, spacing: tuple[float, float, float], step_size: int, smooth: bool = False,
                 smooth_sigma: float = 1.35, max_faces: int = 7500, laplacian_iterations: int = 0) -> dict | None:
    """Tarayıcı sahnesi için topolojisi korunmuş gerçek bir üçgen yüzey üretir."""
    if int(binary.sum()) < MIN_VOXELS:
        return None
    volume = np.pad(binary.astype(np.float32), 1)
    if smooth:
        volume = gaussian_filter(volume, sigma=smooth_sigma)
    # Yüzleri rastgele seyrekleştirmek kapalı yüzeyde delik açar; bunun yerine Marching Cubes
    # örnekleme adımı gerektiğinde uyarlanarak büyütülür, yüzey topolojisi korunur.
    current_step = max(1, step_size)
    while True:
        try:
            vertices, faces, _, _ = marching_cubes(volume, .5, spacing=spacing, step_size=current_step, allow_degenerate=False)
        except ValueError:
            # Çok küçük bir sınıf, görsel yumuşatma sonrası izoyüzeye sahip olmayabilir.
            return None
        # WebGL1/2 ortak yolunda 16-bit indeks güvenliği (indeks taşması şeritli yanlış üçgenler üretir).
        if (len(faces) <= max_faces and len(vertices) <= 60000) or current_step >= 8:
            break
        ratio = max(len(faces) / max_faces, len(vertices) / 60000)
        current_step = min(8, max(current_step + 1, int(np.ceil(current_step * np.sqrt(ratio)))))
    vertices -= np.asarray(spacing)
    # Kullanılmayan tepeler atılır, indeksler sıkıştırılır.
    used_vertices, inverse = np.unique(faces.ravel(), return_inverse=True)
    vertices = vertices[used_vertices]
    faces = inverse.reshape(-1, 3)
    vertices = laplacian_smooth(vertices, faces, iterations=laplacian_iterations)
    return {"vertices": vertices.round(3).tolist(), "faces": faces.tolist()}


def obj_payload(binary: np.ndarray, spacing: tuple[float, float, float]) -> str:
    """Tek katmanın seyrekleştirilmiş OBJ yüzeyi (indirme için)."""
    vertices, faces, _, _ = marching_cubes(np.pad(binary.astype(np.uint8), 1), .5, spacing=spacing,
                                           step_size=2, allow_degenerate=False)
    vertices -= np.asarray(spacing)
    return "\n".join([*(f"v {x} {y} {z}" for x, y, z in vertices), *(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces)])


def scene_payload(record: dict, source: str) -> dict:
    """Masaüstü uygulamasındaki çok-katmanlı 3B sahnenin web karşılığı (beyin yüzeyi + NCR/ED/ET)."""
    mask, spacing = record[source], record["spacing"]
    brain_surface = clean_brain_surface_mask(record["raw"][3])
    layers = {
        # FLAIR'dan gerçek beyin maskesi: küçük arka plan parçaları temizlenir, yeni anatomi üretilmez.
        "brain": {"label": "MR beyin yüzeyi", "color": "#cbd5e1", "opacity": .15,
                  "mesh": mesh_payload(brain_surface, spacing, 1, smooth=True, smooth_sigma=.65, max_faces=52000)},
        "ncr": {"label": "NCR / Nekrotik çekirdek", "color": "#22c55e", "opacity": 1.0, "mesh": mesh_payload(mask == 1, spacing, 1, max_faces=35000)},
        "ed": {"label": "ED / Peritümöral ödem", "color": "#eab308", "opacity": .40, "mesh": mesh_payload(mask == 2, spacing, 1, max_faces=35000)},
        "et": {"label": "ET / Aktif tümör", "color": "#ef4444", "opacity": 1.0, "mesh": mesh_payload(mask == 3, spacing, 1, max_faces=35000)},
    }
    return {"source": source, "layers": layers, "volumes_cm3": core.volumes_cm3(mask, spacing)}
