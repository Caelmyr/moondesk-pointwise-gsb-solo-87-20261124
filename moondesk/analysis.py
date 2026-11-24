"""Measure disk-local intensity and inspect its photo coordinates."""

import numpy as np

from . import frame, model


def section(array, rect):
    x0, y0, x1, y1 = rect
    return array[y0:y1, x0:x1]


def disk_mask(height, width, bounds):
    x0, y0, x1, y1 = bounds
    y, x = np.ogrid[:height, :width]
    u = (2 * (x + 0.5) - x0 - x1) / (x1 - x0)
    v = (2 * (y + 0.5) - y0 - y1) / (y1 - y0)
    return u * u + v * v <= 1


def layers(library, record, settings):
    rgb = frame.pixels(library, record, settings["light"])
    disk = disk_mask(record["height"], record["width"], settings["disk"])
    luma = frame.luma(rgb)
    return rgb, disk, luma, luma >= settings["threshold"]


def percentage(part, total):
    return round(100 * part / total, 2) if total else None


def illumination(disk, bright, settings):
    mask = section(disk, settings["inspect"])
    light = section(bright, settings["inspect"])
    total = int(mask.sum())
    above = int((mask & light).sum())
    arcsec = settings["arcsec_per_pixel"]
    return [{
        "disk_px": total, "light_px": above, "shadow_px": total - above,
        "light_pct": percentage(above, total),
        "light_arcsec2": round(above * arcsec * arcsec, 3),
    }]


def crossings(disk, luma, bright, settings):
    x0, y0, _, _ = settings["inspect"]
    cut_disk = section(disk, settings["inspect"])
    cut_luma = section(luma, settings["inspect"])
    cut_bright = section(bright, settings["inspect"])
    trend = np.diff(cut_luma, axis=1)
    left_sun = settings["sun_side"] == "left"
    if left_sun:
        crossing = cut_bright[:, :-1] & ~cut_bright[:, 1:]
        strength = -trend
    else:
        crossing = ~cut_bright[:, :-1] & cut_bright[:, 1:]
        strength = trend
    valid = crossing & cut_disk[:, :-1] & cut_disk[:, 1:]
    rows = []
    for offset in range(valid.shape[0]):
        candidates = np.flatnonzero(valid[offset])
        if not candidates.size:
            continue
        index = int(candidates[np.argmax(strength[offset, candidates])])
        rows.append({
            "y": y0 + offset, "x": x0 + index + 0.5,
            "jump": round(float(strength[offset, index]), 2), "candidates": int(candidates.size),
        })
    return rows


def profile(luma, disk, bright, settings):
    x0, y0, x1, y1 = settings["line"]
    length = np.hypot(x1 - x0, y1 - y0) * settings["arcsec_per_pixel"]
    n = min(max(abs(x1 - x0), abs(y1 - y0)) + 1, 1000)
    xs = np.rint(np.linspace(x0, x1, n)).astype(int)
    ys = np.rint(np.linspace(y0, y1, n)).astype(int)
    distances = np.linspace(0, length, n)
    return [{
        "x": int(x), "y": int(y), "distance_arcsec": round(float(distance), 3),
        "luma": round(float(luma[y, x]), 2), "in_disk": bool(disk[y, x]), "bright": bool(bright[y, x]),
    } for x, y, distance in zip(xs, ys, distances)]


def reference(luma, settings, key):
    values = section(luma, settings[key])
    quantiles = np.percentile(values, [10, 50, 90])
    return {
        "reference": model.TOOLS[key], "pixels": int(values.size),
        "p10": round(float(quantiles[0]), 2), "median": round(float(quantiles[1]), 2),
        "p90": round(float(quantiles[2]), 2),
    }


def histogram(luma, disk, settings):
    values = section(luma, settings["inspect"])
    inside = section(disk, settings["inspect"])
    bands = list(range(0, 256, 16)) + [256]
    disk_counts, _ = np.histogram(values[inside], bins=bands)
    sky_counts, _ = np.histogram(values[~inside], bins=bands)
    return [{"lower": lo, "upper": hi, "disk_px": int(d), "sky_px": int(s)}
            for lo, hi, d, s in zip(bands, bands[1:], disk_counts, sky_counts)]


def clipping(rgb, disk, settings):
    inside = section(disk, settings["inspect"])
    values = section(rgb, settings["inspect"])[inside]
    rows = []
    for idx, color in enumerate(("红", "绿", "蓝")):
        channel = values[:, idx]
        rows.append({
            "channel": color, "samples": int(channel.size),
            "floor_px": int((channel <= 2).sum()), "ceiling_px": int((channel >= 253).sum()),
        })
    return rows


def point_reading(library, record, settings, point):
    x, y = point
    rgb, disk, luma, bright = layers(library, record, settings)
    return {
        "x": x, "y": y, "rgb": [int(c) for c in rgb[y, x]], "luma": round(float(luma[y, x]), 2),
        "light": settings["light"], "in_disk": bool(disk[y, x]), "bright": bool(bright[y, x]),
    }


def report(library, record, settings, mode):
    rgb, disk, luma, bright = layers(library, record, settings)
    if mode == "illumination":
        rows = illumination(disk, bright, settings)
    elif mode == "crossings":
        rows = crossings(disk, luma, bright, settings)
    elif mode == "profile":
        rows = profile(luma, disk, bright, settings)
    elif mode == "reference":
        rows = [reference(luma, settings, key) for key in ("bright", "dark")]
    elif mode == "histogram":
        rows = histogram(luma, disk, settings)
    else:
        rows = clipping(rgb, disk, settings)
    return {"mode": mode, "heading": model.MODES[mode], "settings": settings, "rows": rows}
