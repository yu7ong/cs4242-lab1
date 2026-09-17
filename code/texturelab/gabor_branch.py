"""Student-facing Gabor branch. Core implementation is intentionally inspectable."""

from __future__ import annotations
from dataclasses import asdict
import numpy as np
from .config import GaborConfig
from .supplied import box_mean, correlate2d


def make_gabor_bank(config: GaborConfig) -> list[tuple[np.ndarray, dict]]:
    """Return ordered zero-mean, unit-norm kernels and their metadata."""
    # YOUR CODE HERE
    #
    # TODO 1 — Validate the configuration
    #   - Require kernel_size to be odd and at least 3. Raise ValueError if not.
    if config.kernel_size % 2 == 0 or config.kernel_size < 3:
        raise ValueError("kernel_size must be odd and at least 3")
    #
    # TODO 2 — Build the centred sampling grid
    #   - Construct float32 x/y coordinates spanning equally on both sides of 0.
    #   - The resulting grid must have shape (kernel_size, kernel_size).
    half = config.kernel_size // 2
    coords = np.arange(-half, half +1, dtype=np.float32)
    x, y = np.meshgrid(coords, coords)

    # TODO 3 — Generate every filter in a stable order
    #   - Loop over frequency first, orientation second, and phase last.
    #   - Rotate the x coordinate into the current orientation.
    #   - Multiply a Gaussian envelope by a sine carrier with the current frequency and phase.
    #   - Remember to add the phase in the metadata.
    bank: list[tuple[np.ndarray, dict]] = []
    for freq in config.frequencies:
        for orient in config.orientations:
            x_rot = x * np.cos(orient) + y * np.sin(orient)
            for phase in config.phases:
                kernel = np.exp(-0.5 * (x**2 + y**2) / (config.sigma**2)) * np.sin(2 * np.pi * freq * x_rot + phase)

                # TODO 4 — Normalise and validate each kernel
                #   - Subtract the kernel mean so constant images have little response.
                #   - Divide by its Euclidean norm and reject a near-zero/degenerate norm.
                #   - Store the final kernel as float32.
                kernel = kernel - kernel.mean()
                norm = np.linalg.norm(kernel)
                if not np.isfinite(norm) or norm < 1e-8:
                    raise ValueError(
                        f"degenerate Gabor kernel (freq={freq}, orient={orient}, phase={phase}): norm={norm}"
                    )
                kernel = (kernel / norm).astype(np.float32)

                # TODO 5 — Attach metadata and return
                #   - Include frequency, orientation, phase, and asdict(config) for each item.
                #   - Return a list of (kernel, metadata) tuples in the loop order above.
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
    #
    # TODO 1 — Compute one response map per bank entry
    #   - Preserve bank order and correlate (do not convolve) gray with each kernel.
    channels = []
    for kernel, _ in bank:
        response = correlate2d(gray, kernel)

        # TODO 2 — Convert responses to non-negative energy
        #   - For "squared", square the signed response.
        #   - For "absolute", take its absolute value.
        #   - Raise ValueError for any other energy name.
        if energy == "squared":
            energy_map = response ** 2
        elif energy == "absolute":
            energy_map = np.abs(response)
        else:
            raise ValueError(f"unknown energy mode: {energy!r}")

        # TODO 3 — Pool and assemble the channels
        #   - Apply box_mean with pool_size to every energy map.
        #   - Pooling must preserve the original image height and width.
        pooled = box_mean(energy_map, pool_size)
        channels.append(pooled)

    #   - Stack maps on the final axis to obtain H x W x K float32 output.
    result = np.stack(channels, axis=-1).astype(np.float32)

    # TODO 4 — Check numerical validity
    #   - Raise FloatingPointError if any returned value is NaN or infinite.
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("gabor_energy_maps produced non-finite values")

    return result

if __name__ == "__main__":
    # Quick sanity check of the Gabor bank generation
    config = GaborConfig()
    bank = make_gabor_bank(config)
    print(f"Generated {len(bank)} Gabor kernels with shape {bank[0][0].shape} and dtype {bank[0][0].dtype}")
    print(bank[0][1])  # Print metadata for the first kernel
