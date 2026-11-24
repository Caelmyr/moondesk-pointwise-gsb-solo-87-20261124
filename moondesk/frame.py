"""Decode single-frame exposures and derive their displayed luma."""

from io import BytesIO
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from image_lab.validation import Invalid

from .model import MAX_PIXELS, MAX_UPLOAD


def encode(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def decode(raw):
    if len(raw) > MAX_UPLOAD:
        raise Invalid("单张输入最大为 14 MiB。")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(raw), formats=("PNG", "JPEG")) as source:
                width, height = source.size
                if min(width, height) < 8 or width * height > MAX_PIXELS:
                    raise Invalid("图像的两边至少 8 像素，总像素不能超过 400 万。")
                if getattr(source, "n_frames", 1) != 1:
                    raise Invalid("请使用单帧 PNG 或 JPEG。")
                prepared = ImageOps.exif_transpose(source)
                if "A" in prepared.getbands() or "transparency" in prepared.info:
                    if prepared.convert("RGBA").getchannel("A").getextrema()[0] != 255:
                        raise Invalid("输入须先铺好底色并去除透明像素。")
                image = prepared.convert("RGB")
    except Invalid:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise Invalid("图片超过可处理的分辨率。") from None
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        raise Invalid("无法打开图像。请检查 PNG 或 JPEG 文件。") from None
    return image


def pixels(library, record, light):
    with Image.open(library.source(record, light)) as sample:
        return np.array(sample.convert("RGB"))


def luma(rgb):
    colors = rgb.astype(np.float32)
    return colors[..., 0] * 0.2126 + colors[..., 1] * 0.7152 + colors[..., 2] * 0.0722
