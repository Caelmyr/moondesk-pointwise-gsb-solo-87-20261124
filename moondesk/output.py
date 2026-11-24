"""Render source-space annotations and export measured rows."""

import csv
from io import StringIO

import numpy as np
from PIL import Image, ImageDraw

from . import analysis, frame


def csv_text(report):
    target = StringIO(newline="")
    writer = csv.writer(target)
    writer.writerow(("mode", report["mode"], "source", report["settings"]["light"]))
    chosen = report["settings"]
    writer.writerow(("x0", "y0", "x1", "y1", "threshold", "arcsec_per_pixel", "sun_side"))
    writer.writerow((*chosen["inspect"], chosen["threshold"], chosen["arcsec_per_pixel"], chosen["sun_side"]))
    if report["rows"]:
        keys = list(report["rows"][0])
        writer.writerow(keys)
        writer.writerows([row.get(key) for key in keys] for row in report["rows"])
    return target.getvalue()


def overlay(library, record, report):
    chosen = report["settings"]
    rgb, disk, _, bright = analysis.layers(library, record, chosen)
    mode = report["mode"]
    marks = np.zeros(disk.shape + (4,), dtype=np.uint8)
    marked = analysis.section(marks, chosen["inspect"])
    in_disk = analysis.section(disk, chosen["inspect"])
    illuminated = analysis.section(bright, chosen["inspect"])
    if mode == "illumination":
        marked[in_disk & illuminated] = (250, 199, 122, 80)
    if mode == "clipping":
        clipped = ((analysis.section(rgb, chosen["inspect"]) >= 253).any(axis=2))
        marked[in_disk & clipped] = (244, 122, 166, 190)
    image = Image.alpha_composite(Image.fromarray(rgb).convert("RGBA"), Image.fromarray(marks))
    draw = ImageDraw.Draw(image)
    for key, color in (("disk", "#e8be82"), ("inspect", "#96d2ce")):
        x0, y0, x1, y1 = chosen[key]
        shape = draw.ellipse if key == "disk" else draw.rectangle
        shape((x0, y0, x1 - 1, y1 - 1), outline=color, width=1)
    if mode == "profile":
        draw.line(chosen["line"], fill="#96d2ce", width=2)
    if mode == "reference":
        for key in ("bright", "dark"):
            draw.rectangle(chosen[key], outline="#e8be82", width=2)
    if mode == "crossings":
        for row in report["rows"]:
            x, y = row["x"], row["y"]
            draw.line((x - 1, y, x + 1, y), fill="#f8bc81", width=1)
    return frame.encode(image.convert("RGB"))
