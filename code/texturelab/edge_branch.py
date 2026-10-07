"""Student-facing exact-direction NMS, adaptive thresholds, and hysteresis."""

from __future__ import annotations
from collections import deque
import numpy as np
from .config import EdgeConfig
from .supplied import gaussian_smooth, sobel_gradients


def bilinear_sample(image: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Sample a 2-D image at floating-point image coordinates. [AI-Code] [Human-Check]"""
    H, W = image.shape
    y = np.clip(y, 0, H - 1)
    x = np.clip(x, 0, W - 1)
    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    y1 = np.clip(y0 + 1, 0, H - 1)
    x1 = np.clip(x0 + 1, 0, W - 1)

    wy = y - y0 
    wx = x - x0
    top_left     = image[y0, x0]
    top_right    = image[y0, x1]
    bottom_left  = image[y1, x0]
    bottom_right = image[y1, x1]

    top    = top_left * (1 - wx) + top_right * wx
    bottom = bottom_left * (1 - wx) + bottom_right * wx
    result = top * (1 - wy) + bottom * wy
    return result


def nms_interpolated(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Thin gradient magnitude along the exact image-coordinate direction. [AI-Code] [Human-Check]"""

    if magnitude.shape != direction.shape or magnitude.ndim != 2:
        raise ValueError("magnitude and direction must be matching 2-D arrays")

    h, w = magnitude.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)

    dy = np.sin(direction)
    dx = np.cos(direction)

    mag_fwd = bilinear_sample(magnitude, ys + dy, xs + dx)
    mag_bwd = bilinear_sample(magnitude, ys - dy, xs - dx)

    keep = (magnitude >= mag_fwd) & (magnitude >= mag_bwd)
    result = np.where(keep, magnitude, 0.0).astype(np.float32)

    result[0, :] = 0
    result[-1, :] = 0
    result[:, 0] = 0
    result[:, -1] = 0

    return result


def adaptive_thresholds(nms: np.ndarray, config: EdgeConfig) -> tuple[float, float]:
    """Select reproducible low/high thresholds without test labels. [AI-Code] [Human-Check]"""

    positive = nms[nms > 0]
    if positive.size == 0:
        return (float("inf"), float("inf"))
    
    if config.threshold_method == "percentile":
        high = np.percentile(positive, config.high_percentile)
    elif config.threshold_method == "robust":
        median = np.median(positive)
        mad = np.median(np.abs(positive - median))
        high = median + 2.5 * 1.4826 * mad
    else:
        raise ValueError(f"unknown threshold method: {config.threshold_method}")

    low = config.low_ratio * high
    return (float(low), float(high))


def hysteresis(strong: np.ndarray, weak: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """Keep all weak pixels reachable from strong seeds. [AI-Code] [Human-Check]"""

    if strong.shape != weak.shape:
        raise ValueError("strong and weak masks must have matching shapes")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")


    strong_bool = strong.astype(bool)
    weak_bool = weak.astype(bool)

    H, W = strong_bool.shape
    result = strong_bool.copy()
    queue = deque(map(tuple, np.argwhere(strong_bool)))


    if connectivity == 4:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while queue:
        r, c = queue.popleft()
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                if weak_bool[nr, nc] and not result[nr, nc]:
                    result[nr, nc] = True
                    queue.append((nr, nc))

    return result


def detect_edges(gray: np.ndarray, config: EdgeConfig) -> dict[str, np.ndarray | float]:
    """Run the supplied smoothing/Sobel stages and student edge stages. [AI-Code] [Human-Check]"""
    smoothed = gaussian_smooth(gray, config.gaussian_sigma)
    gx, gy, magnitude, direction = sobel_gradients(smoothed)

    nms = nms_interpolated(magnitude, direction)
    low, high = adaptive_thresholds(nms, config)
    strong = nms >= high
    weak = (nms >= low) & ~strong

    edges = hysteresis(strong, weak, config.connectivity)
    return {
        "gx": gx,
        "gy": gy,
        "magnitude": magnitude,
        "direction": direction,
        "nms": nms,
        "low": low,
        "high": high,
        "edges": edges,
    }
