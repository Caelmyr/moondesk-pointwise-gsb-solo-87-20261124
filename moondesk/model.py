"""Validate capture geometry, exposure ports, and inspection settings."""

import math

from image_lab.validation import Invalid

MAX_UPLOAD = 14 * 1024 * 1024
MAX_PIXELS = 4_000_000
LIGHTS = {"a": "主照片", "b": "对照照片"}
MODES = {
    "illumination": "受光统计",
    "crossings": "明暗横扫",
    "profile": "亮度剖线",
    "reference": "亮暗参照",
    "histogram": "亮度分布",
    "clipping": "剪切检查",
}
TOOLS = {
    "inspect": "检查窗",
    "disk": "月盘范围",
    "bright": "亮区参照",
    "dark": "暗区参照",
    "line": "剖线",
}


def document(value):
    if not isinstance(value, dict):
        raise Invalid("请提供检查设置。")
    return value


def text(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 160:
        raise Invalid("名称应有 1 至 160 个字符。")
    return value.strip()


def number(value, name, lower, upper):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Invalid(f"{name}应是数字。")
    if not math.isfinite(value) or not lower <= value <= upper:
        raise Invalid(f"{name}须在 {lower} 至 {upper} 之间。")
    return value


def positions(value, size, rectangle):
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise Invalid("请在图上选择两个点。")
    for coordinate, bound in zip(value, (*size, *size)):
        if type(coordinate) is not int or not 0 <= coordinate < bound + rectangle:
            raise Invalid("请选择源图范围内的整像素坐标。")
    x0, y0, x1, y1 = value
    if rectangle and (x0 >= x1 or y0 >= y1):
        raise Invalid("检查框必须有正宽和正高。")
    return [x0, y0, x1, y1]


def defaults(width, height):
    return {
        "light": "a", "threshold": 75, "arcsec_per_pixel": 2, "sun_side": "left",
        "inspect": [0, 0, width, height],
        "disk": [width // 8, height // 8, width - width // 8, height - height // 8],
        "bright": [width // 4, height // 3, width // 2, height * 2 // 3],
        "dark": [width * 2 // 3, height // 3, width * 3 // 4, height * 2 // 3],
        "line": [0, height // 2, width - 1, height // 2],
    }


def settings(value, panel):
    value = document(value)
    base = panel["settings"]
    size = panel["width"], panel["height"]
    chosen = {key: positions(value.get(key, base[key]), size, key != "line") for key in TOOLS}
    light = value.get("light", base["light"])
    if not isinstance(light, str) or light not in panel["sources"]:
        raise Invalid("尚未导入该照片。")
    sun = value.get("sun_side", base["sun_side"])
    if sun not in ("left", "right"):
        raise Invalid("请选择月面亮侧的大致方向。")
    chosen.update(
        light=light, sun_side=sun,
        threshold=number(value.get("threshold", base["threshold"]), "亮度门槛", 0, 255),
        arcsec_per_pixel=number(value.get("arcsec_per_pixel", base["arcsec_per_pixel"]), "角秒每像素", 0.001, 1200),
    )
    return chosen


def mode(value):
    if not isinstance(value, str) or value not in MODES:
        raise Invalid("请选择一个检查项目。")
    return value


def point(value, panel):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise Invalid("请在图面上取样。")
    x, y, _, _ = positions((*value, *value), (panel["width"], panel["height"]), False)
    return x, y
