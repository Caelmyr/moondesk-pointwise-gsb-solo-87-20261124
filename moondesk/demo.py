"""Make labeled synthetic observations with craters, earthshine and clipping."""

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from . import model


def create(library):
    width, height = 820, 580
    y, x = np.mgrid[:height, :width]
    u = (x - 410) / 223
    v = (y - 290) / 223
    disk = u * u + v * v <= 1
    bulge = np.sqrt(np.maximum(0, 1 - u ** 2 - v ** 2))
    terminator = 0.23 * (1 - v ** 2) + 0.035 * np.sin(11 * v)
    light = np.clip((terminator - u) * 24 + 0.5, 0, 1)
    random = np.random.default_rng(61423)
    noise = gaussian_filter(random.normal(size=(height, width)), 2) * 54
    terrain = 0.5 * np.sin(11 * u + 3 * v) + 0.35 * np.cos(9 * v - 4 * u)
    relief = np.zeros((height, width))
    for cx, cy, radius in random.uniform([-0.9, -0.9, 0.017], [0.7, 0.9, 0.16], (38, 3)):
        radial = np.hypot(u - cx, v - cy)
        rim = np.exp(-((radial - radius) / (radius * 0.16)) ** 2)
        shadow = np.exp(-(((u - cx - radius * 0.34) / radius) ** 2 + ((v - cy) / radius) ** 2) * 3)
        relief += 21 * rim - 40 * shadow
    moon = 18 + (light * (75 + 91 * bulge + terrain * 41 + relief)) + noise
    sky = np.broadcast_to(6 + y * 0.003, (height, width)).copy()
    for sx, sy in random.integers([6, 6], [width - 6, height - 6], (41, 2)):
        sky[sy, sx] = random.integers(35, 110)
    luminance = np.where(disk, moon, sky + noise * 0.17)
    sources = {}
    for role, factor in (("a", 1), ("b", 1.78)):
        level = gaussian_filter(luminance, 0.58) * factor
        rgb = np.stack([level * 1.04, level, level * 0.92 + 3], axis=-1)
        photo = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")
        sources[role] = photo, f"synthetic-moon-{role}.png"
    settings = model.defaults(width, height)
    settings.update(disk=[187, 67, 634, 514], inspect=[185, 67, 636, 514],
                    bright=[255, 206, 325, 272], dark=[515, 244, 580, 330],
                    line=[204, 290, 624, 290], arcsec_per_pixel=4.06)
    return library.create("弦月样本 · 合成练习", sources, settings, synthetic=True)
