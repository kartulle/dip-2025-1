# histogram_matching_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `match_histograms_rgb(source_img, reference_img)` that receives two RGB images
(as NumPy arrays with shape (H, W, 3)) and returns a new image where the histogram of each RGB channel 
from the source image is matched to the corresponding histogram of the reference image.

Your task:
- Read two RGB images: source and reference (they will be provided externally).
- Match the histograms of the source image to the reference image using all RGB channels.
- Return the matched image as a NumPy array (uint8)

Function signature:
    def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray

Return:
    - matched_img: NumPy array of the result image

Notes:
- Do NOT save or display the image in this function.
- Do NOT use OpenCV to apply the histogram match (only for loading images, if needed externally).
- You can assume the input images are already loaded and in RGB format (not BGR).
"""

import cv2 as cv
import numpy as np

def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:

    def _match_channel(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
        src_u8 = np.clip(np.rint(src).astype(np.int32), 0, 255)
        ref_u8 = np.clip(np.rint(ref).astype(np.int32), 0, 255)

        src_hist = np.bincount(src_u8.ravel(), minlength=256).astype(np.float64)
        ref_hist = np.bincount(ref_u8.ravel(), minlength=256).astype(np.float64)

        src_cdf = np.cumsum(src_hist); src_cdf /= src_cdf[-1] if src_cdf[-1] > 0 else 1.0
        ref_cdf = np.cumsum(ref_hist); ref_cdf /= ref_cdf[-1] if ref_cdf[-1] > 0 else 1.0

        lut = np.searchsorted(ref_cdf, src_cdf, side="left")
        lut = np.clip(lut, 0, 255).astype(np.uint8)

        return lut[src_u8].astype(np.uint8)

    src = source_img if source_img.dtype == np.uint8 else np.clip(np.rint(source_img), 0, 255).astype(np.uint8)
    ref = reference_img if reference_img.dtype == np.uint8 else np.clip(np.rint(reference_img), 0, 255).astype(np.uint8)

    matched = np.empty_like(src, dtype=np.uint8)
    for c in range(3):
        matched[..., c] = _match_channel(src[..., c], ref[..., c])

    return matched
