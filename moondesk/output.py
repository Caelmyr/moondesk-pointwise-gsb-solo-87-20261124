"""Render source-space annotations and export measured rows."""

import csv
from io import StringIO

import numpy as np
from PIL import Image, ImageDraw

from . import analysis, boundary, frame

LIGHT_FILL = (250, 199, 122, 80)
DARK_FILL = (72, 92, 168, 96)
PENDING_FILL = (238, 70, 180, 150)
EDGE = "#96f2e6"
PRIMARY = "#ffffff"
SPUR = "#f8a94f"
BLANK = "#ff6f86"


def csv_text(report):
    target = StringIO(newline="")
    writer = csv.writer(target)
    writer.writerow(("mode", report["mode"], "source", report["settings"]["light"]))
    chosen = report["settings"]
    writer.writerow(("x0", "y0", "x1", "y1", "threshold", "arcsec_per_pixel",
                     "sun_side", "boundary"))
    writer.writerow((*chosen["inspect"], chosen["threshold"], chosen["arcsec_per_pixel"],
                     chosen["sun_side"], chosen.get("boundary", "trace")))
    if report.get("trace"):
        trace = report["trace"]
        writer.writerow(("boundary_summary", trace["summary"]))
        for segment in trace["segments"]:
            writer.writerow(("segment", segment["y0"], segment["y1"], segment["rows"],
                             segment["mean_jump"], segment["max_step"]))
        for segment in trace["alternatives"]:
            writer.writerow(("unused_deflection", segment["y0"], segment["y1"], segment["rows"],
                             segment["mean_jump"], segment["offset"]))
    if report["rows"]:
        keys = list(report["rows"][0])
        writer.writerow(keys)
        writer.writerows([row.get(key) for key in keys] for row in report["rows"])
    return target.getvalue()


def _boundary_marks(marks, disk, settings, trace):
    """Paint light/dark/pending areas behind the trace and ring blank rows."""
    labels = boundary.labels(disk, settings, trace["points"])
    inside = boundary.window(settings, disk.shape)
    marked = analysis.section(marks, settings["inspect"])
    cut = analysis.section(labels, settings["inspect"])
    cut_disk = analysis.section(disk & inside, settings["inspect"])
    marked[cut_disk & (cut == boundary.LIGHT)] = LIGHT_FILL
    marked[cut_disk & (cut == boundary.DARK)] = DARK_FILL
    marked[cut_disk & (cut == boundary.PENDING)] = PENDING_FILL


def _draw_boundary(draw, trace, top):
    primary = trace["primary"]
    for segment in trace["segments"]:
        width = 2 if segment is primary else 1
        draw.line([(x, y) for y, x in segment["points"]], fill=EDGE, width=width)
        for y, x in segment["points"]:
            draw.point((x, y), fill=EDGE)
    for segment in trace["alternatives"]:
        draw.line([(x, y) for y, x in segment["points"]], fill=SPUR, width=2)
        for y, x in segment["points"]:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline=SPUR, width=1)
    for row in trace["rows"]:
        if row["x"] is None:
            y = row["y"]
            draw.ellipse((top + 2, y - 2, top + 8, y + 2), outline=BLANK, width=1)


def overlay(library, record, report):
    chosen = report["settings"]
    rgb, disk, _, bright = analysis.layers(library, record, chosen)
    mode = report["mode"]
    marks = np.zeros(disk.shape + (4,), dtype=np.uint8)
    marked = analysis.section(marks, chosen["inspect"])
    in_disk = analysis.section(disk, chosen["inspect"])
    illuminated = analysis.section(bright, chosen["inspect"])
    trace = report.get("trace")
    if mode == "illumination" and trace:
        _boundary_marks(marks, disk, chosen, trace)
    elif mode == "illumination":
        marked[in_disk & illuminated] = LIGHT_FILL
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
    if trace and mode in ("illumination", "crossings"):
        _draw_boundary(draw, trace, chosen["inspect"][0])
    return frame.encode(image.convert("RGB"))
