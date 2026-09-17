"""Student-facing Gabor branch. Core implementation is intentionally inspectable."""

from __future__ import annotations
from dataclasses import asdict
import numpy as np
from .config import GaborConfig
from .supplied import box_mean, correlate2d


def make_gabor_bank(config: GaborConfig) -> list[tuple[np.ndarray, dict]]:
    """Return ordered zero-mean, unit-norm kernels and their metadata."""

    if config.kernel_size % 2 == 0 or config.kernel_size < 3:
        raise ValueError("kernel_size must be odd and at least 3")

    half = config.kernel_size // 2
    coords = np.arange(-half, half +1, dtype=np.float32)
    x, y = np.meshgrid(coords, coords)

    bank: list[tuple[np.ndarray, dict]] = []
    for freq in config.frequencies:
        for orient in config.orientations:
            x_rot = x * np.cos(orient) + y * np.sin(orient)
            for phase in config.phases:
                kernel = np.exp(-0.5 * (x**2 + y**2) / (config.sigma**2)) * np.sin(2 * np.pi * freq * x_rot + phase)

                kernel = kernel - kernel.mean()
                norm = np.linalg.norm(kernel)
                if not np.isfinite(norm) or norm < 1e-8:
                    raise ValueError(
                        f"degenerate Gabor kernel (freq={freq}, orient={orient}, phase={phase}): norm={norm}"
                    )
                kernel = (kernel / norm).astype(np.float32)

                metadata = {
                    "frequency": freq,
                    "orientation": orient,
                    "phase": phase,
                    "config": asdict(config),
                }
                bank.append((kernel, metadata))
    return bank


def gabor_energy_maps(gray: np.ndarray, bank: list[tuple[np.ndarray, dict]], pool_size: int = 9,
                      energy: str = "squared") -> np.ndarray:
    """Return finite H x W x K locally pooled energy maps."""

    channels = []
    for kernel, _ in bank:
        response = correlate2d(gray, kernel)

        if energy == "squared":
            energy_map = response ** 2
        elif energy == "absolute":
            energy_map = np.abs(response)
        else:
            raise ValueError(f"unknown energy mode: {energy!r}")

        pooled = box_mean(energy_map, pool_size)
        channels.append(pooled)

    result = np.stack(channels, axis=-1).astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("gabor_energy_maps produced non-finite values")

    return result
