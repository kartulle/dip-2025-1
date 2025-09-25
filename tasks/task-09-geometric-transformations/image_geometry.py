# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np

def apply_geometric_transformations(img: np.ndarray) -> dict:
    h, w = img.shape
    in_dtype = img.dtype

    dy = max(1, int(round(0.1 * h)))
    dx = max(1, int(round(0.1 * w)))
    translated = np.zeros_like(img)
    translated[dy:, dx:] = img[:h - dy, :w - dx]

    rotated = np.rot90(img, k=-1)

    scale = 1.5
    new_w = int(np.ceil(w * scale))
    x_dst = np.arange(new_w) / scale
    x0 = np.floor(x_dst).astype(int)
    x1 = np.clip(x0 + 1, 0, w - 1)
    alpha = x_dst - x0
    stretched = ((1 - alpha) * img[:, x0] + alpha * img[:, x1]).astype(in_dtype)

    mirrored = img[:, ::-1]

    k = -0.1
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    sx, sy = max(cx, (w - 1) - cx) or 1.0, max(cy, (h - 1) - cy) or 1.0
    x_dn = (xx - cx) / sx
    y_dn = (yy - cy) / sy
    r2 = x_dn*2 + y_dn*2

    denom = 1.0 + k * r2
    x_un = x_dn / denom
    y_un = y_dn / denom

    x_src = (x_un * sx + cx).round().astype(int)
    y_src = (y_un * sy + cy).round().astype(int)

    distorted = np.zeros_like(img)
    ok = (x_src >= 0) & (x_src < w) & (y_src >= 0) & (y_src < h)
    distorted[ok] = img[y_src[ok], x_src[ok]]

    return {
        "translated": translated,
        "rotated": rotated,
        "stretched": stretched,
        "mirrored": mirrored,
        "distorted": distorted
    }