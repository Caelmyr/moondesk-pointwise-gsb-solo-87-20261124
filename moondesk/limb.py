"""Fit a circular lunar limb from the aligned short and long exposures."""

import math

import numpy as np
from scipy.ndimage import map_coordinates

from image_lab.validation import Invalid

from . import frame

RAY_STEP_DEG = 2.0
CUSP_MARGIN_DEG = 25.0
WINDOW_FRACTION = 0.12
WINDOW_MIN_PX = 8.0
SAMPLE_STEP = 0.5
ARC_STEP_DEG = 2.0
ARC_SPAN_DEG = 8.0
MIN_POINTS = 12


def sky_sigma(luma):
    """Estimate sky noise from the darkest corner patch."""
    height, width = luma.shape
    side = max(4, min(16, height // 4, width // 4))
    corners = [luma[y:y + side, x:x + side] for y in (0, height - side) for x in (0, width - side)]
    patch = min(corners, key=lambda part: float(np.median(part)))
    median = float(np.median(patch))
    return max(1.4826 * float(np.median(np.abs(patch - median))), 1.0)


def box_limit(center, dx, dy, width, height):
    """Distance from center to the image edge along a unit direction."""
    limit = math.inf
    if dx > 0:
        limit = (width - 1 - center[0]) / dx
    elif dx < 0:
        limit = -center[0] / dx
    if dy > 0:
        limit = min(limit, (height - 1 - center[1]) / dy)
    elif dy < 0:
        limit = min(limit, -center[1] / dy)
    return limit


def ray_edge(luma, center, angle, expected, window, sigma):
    """Locate the strongest sustained outward brightness drop along one ray.

    The profile is averaged over a small tangential arc so that noise and
    surface texture cancel while the limb step stays coherent.
    """
    height, width = luma.shape
    spread = np.radians(np.arange(-ARC_SPAN_DEG, ARC_SPAN_DEG + 1e-9, ARC_STEP_DEG))
    cosines = np.cos(angle + spread)
    sines = np.sin(angle + spread)
    limit = min(box_limit(center, float(c), float(s), width, height) for c, s in zip(cosines, sines))
    r_lo = max(expected - window, 2.0)
    r_hi = min(expected + window, limit - 1.0)
    if r_hi - r_lo < 6.0:
        return None
    radii = np.arange(r_lo, r_hi, SAMPLE_STEP)
    xs = center[0] + np.outer(cosines, radii)
    ys = center[1] + np.outer(sines, radii)
    sampled = map_coordinates(luma, (ys.ravel(), xs.ravel()), order=1, mode="nearest",
                              prefilter=False).reshape(xs.shape)
    values = sampled.mean(axis=0)
    step = max(1, round(1.0 / SAMPLE_STEP))
    margin = int(round(4.0 / SAMPLE_STEP))
    if values.size < 2 * (margin + step) + 1:
        return None
    integral = np.concatenate(([0.0], np.cumsum(values)))
    idx = np.arange(margin + step - 1, values.size - margin - step)
    in_start = idx - margin - step + 1
    in_end = idx - step + 1
    out_start = idx + step
    out_end = idx + step + margin
    sustained = (integral[in_end] - integral[in_start]
                 - (integral[out_end] - integral[out_start])) / margin
    pick = int(np.argmax(sustained))
    if sustained[pick] < max(3.0 * sigma, 3.0):
        return None
    index = int(idx[pick])
    offset = 0.0
    if 0 < pick < sustained.size - 1:
        left, peak, right = sustained[pick - 1], sustained[pick], sustained[pick + 1]
        denom = float(left - 2.0 * peak + right)
        if denom > 1e-9:
            offset = float(np.clip(0.5 * (left - right) / denom, -1.0, 1.0))
    radius = float(radii[index]) + offset * SAMPLE_STEP
    return center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle)


def fit_circle(points):
    x = points[:, 0]
    y = points[:, 1]
    system = np.column_stack((x, y, np.ones(x.size)))
    solution, *_ = np.linalg.lstsq(system, -(x * x + y * y), rcond=None)
    cx, cy = float(-solution[0] / 2.0), float(-solution[1] / 2.0)
    radius = math.sqrt(max(cx * cx + cy * cy - float(solution[2]), 1e-9))
    return cx, cy, radius


def deviation(points, circle):
    cx, cy, radius = circle
    return np.hypot(points[:, 0] - cx, points[:, 1] - cy) - radius


def robust_fit(points, rounds=4):
    keep = np.ones(points.shape[0], dtype=bool)
    circle = fit_circle(points)
    for _ in range(rounds):
        circle = fit_circle(points[keep])
        spread = 1.4826 * float(np.median(np.abs(deviation(points[keep], circle))))
        gate = max(2.5 * spread, 1.25)
        updated = np.abs(deviation(points, circle)) <= gate
        if (updated == keep).all():
            break
        keep = updated
    return circle, keep


def angle_arcs(angles, gap=RAY_STEP_DEG * 2.6):
    if not angles.size:
        return []
    ordered = np.sort(angles)
    cuts = np.flatnonzero(np.diff(ordered) > gap) + 1
    return [[float(group[0]), float(group[-1])] for group in np.split(ordered, cuts)]


def propose(library, record, settings):
    if "b" not in record["sources"]:
        raise Invalid("拟合圆形月缘需要主照片和对照照片两张已对齐的曝光。")
    lumas = {role: frame.luma(frame.pixels(library, record, role)) for role in ("a", "b")}
    x0, y0, x1, y1 = settings["bright"]
    clipped = {role: float((luma[y0:y1, x0:x1] >= 253).mean()) for role, luma in lumas.items()}
    short = min(clipped, key=lambda role: (clipped[role], role != "a"))
    long = "b" if short == "a" else "a"
    height, width = record["height"], record["width"]
    dx0, dy0, dx1, dy1 = settings["disk"]
    center = ((dx0 + dx1) / 2.0, (dy0 + dy1) / 2.0)
    semi = ((dx1 - dx0) / 2.0, (dy1 - dy0) / 2.0)
    sun_deg = 180.0 if settings["sun_side"] == "left" else 0.0
    middles = {"bright": sun_deg, "dark": (sun_deg + 180.0) % 360.0}
    images = {"bright": lumas[short], "dark": lumas[long]}
    lights = {"bright": short, "dark": long}
    half = math.radians(90.0 - CUSP_MARGIN_DEG)
    points = []
    for side, middle_deg in middles.items():
        luma = images[side]
        sigma = sky_sigma(luma)
        middle = math.radians(middle_deg)
        angle = middle - half
        while angle <= middle + half + 1e-9:
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            expected = semi[0] * semi[1] / math.hypot(semi[1] * cos_a, semi[0] * sin_a)
            window = max(WINDOW_FRACTION * expected, WINDOW_MIN_PX)
            edge = ray_edge(luma, center, angle, expected, window, sigma)
            if edge:
                points.append((edge[0], edge[1], side))
            angle += math.radians(RAY_STEP_DEG)
    if len(points) < MIN_POINTS:
        raise Invalid("在月盘框附近没有找到足够的月缘点。请检查月盘框位置。")
    xy = np.array([(point[0], point[1]) for point in points])
    circle, keep = robust_fit(xy)
    if int(keep.sum()) < MIN_POINTS:
        raise Invalid("月缘点过于分散，无法拟合出稳定的圆。")
    cx, cy, radius = circle
    if not (0.0 <= cx <= width and 0.0 <= cy <= height and 8.0 <= radius <= 2.0 * max(width, height)):
        raise Invalid("拟合结果超出画面。请调整月盘框后重试。")
    accepted = [point for point, ok in zip(points, keep) if ok]
    rms_all = float(np.sqrt(np.mean(deviation(xy[keep], circle) ** 2)))
    if rms_all > 3.0:
        raise Invalid(f"在月盘框附近找不到一致的圆形月缘（偏差 RMS {rms_all:.1f} px）。请调整月盘框后重试。")

    def side_report(side):
        chosen = np.array([(p[0], p[1]) for p in accepted if p[2] == side])
        report = {"light": lights[side], "arcs": [], "count": 0, "rms_px": None, "max_px": None}
        if not chosen.size:
            return report
        dev = np.abs(deviation(chosen, circle))
        angles = np.degrees(np.arctan2(chosen[:, 1] - cy, chosen[:, 0] - cx))
        middle = middles[side]
        relative = (angles - middle + 180.0) % 360.0 - 180.0
        report.update(
            arcs=[[int(round(a + middle)), int(round(b + middle))] for a, b in angle_arcs(relative)],
            count=int(chosen.shape[0]),
            rms_px=round(float(np.sqrt(np.mean(dev ** 2))), 3),
            max_px=round(float(dev.max()), 3),
        )
        return report

    def single_fit(side):
        chosen = np.array([(p[0], p[1]) for p in accepted if p[2] == side])
        if chosen.shape[0] < 8:
            return None
        angles = np.degrees(np.arctan2(chosen[:, 1] - cy, chosen[:, 0] - cx))
        relative = (angles - middles[side] + 180.0) % 360.0 - 180.0
        if float(relative.max() - relative.min()) < 40.0:
            return None
        fitted, kept = robust_fit(chosen, rounds=2)
        return fitted if int(kept.sum()) >= 8 else None

    bright_fit = single_fit("bright")
    dark_fit = single_fit("dark")
    conflict = None
    notes = []
    if bright_fit and dark_fit:
        conflict = {
            "center_shift_px": round(math.hypot(bright_fit[0] - dark_fit[0], bright_fit[1] - dark_fit[1]), 2),
            "radius_gap_px": round(abs(bright_fit[2] - dark_fit[2]), 2),
        }
    else:
        notes.append("某一侧可用月缘不足，未给出两侧各自拟合的对比。")
    left = int(round(cx - radius))
    top = int(round(cy - radius))
    right = int(round(cx + radius))
    bottom = int(round(cy + radius))
    if left < 0 or top < 0 or right > width or bottom > height:
        notes.append("月缘圆超出画面，建议框已裁到图内。")
    disk = [min(max(left, 0), width - 2), min(max(top, 0), height - 2),
            max(min(right, width), 2), max(min(bottom, height), 2)]
    samples = [[round(float(p[0]), 1), round(float(p[1]), 1), p[2]] for p in accepted]
    return {
        "circle": {"cx": round(cx, 2), "cy": round(cy, 2), "radius": round(radius, 2)},
        "disk": disk,
        "rms_px": round(rms_all, 3),
        "samples": samples,
        "sides": {"bright": side_report("bright"), "dark": side_report("dark")},
        "conflict": conflict,
        "notes": notes,
    }
