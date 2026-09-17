"""Aligned local descriptor shared by all three tasks."""

from __future__ import annotations
import numpy as np
from .config import FeatureConfig
from .edge_branch import detect_edges
from .gabor_branch import gabor_energy_maps, make_gabor_bank
from .supplied import box_mean, rgb2gray


def extract_local_features(image: np.ndarray, config: FeatureConfig) -> tuple[np.ndarray, list[str]]:
    """Return an aligned H x W x D map and one name per channel."""
    # YOUR CODE HERE
    #
    # TODO 1 — Normalise the input representation
    #   - Convert to float32; expand a 2-D image to three identical channels.
    #   - Scale 0--255-style inputs to 0--1 while leaving 0--1 inputs unchanged.
    #   - Compute grayscale once with rgb2gray.
    #
    # TODO 2 — Initialise aligned channel/name groups
    #   - Keep feature chunks and their names in matching, deterministic order.
    #   - If include_colour, append the first three image channels named
    #     colour_r, colour_g, and colour_b.
    #
    # TODO 3 — Add the Gabor branch when enabled
    #   - Build config.gabor's bank and compute pooled maps using its pool_size
    #     and energy mode.
    #   - Create one stable name from each kernel's frequency/orientation/phase.
    #
    # TODO 4 — Compute edge data only when needed
    #   - Call detect_edges once if gradient or edge channels are requested.
    #   - For gradient energy, square magnitude, box-pool it with density_size,
    #     add a singleton channel axis, and name it gradient_energy.
    #
    # TODO 5 — Add total and orientation-specific edge density
    #   - Pool the Boolean edge map for edge_density.
    #   - Map directions modulo pi, divide the unsigned orientation range into
    #     config.edge.n_orientations bins, and pool edges belonging to each bin.
    #   - Name bins edge_orientation_0, edge_orientation_1, and so on.
    #
    # TODO 6 — Assemble and optionally standardise
    #   - Reject a configuration with no enabled feature family.
    #   - Concatenate chunks along the final axis and return float32.
    #   - If standardise_per_image, standardise every channel over H/W using a
    #     small epsilon; return the feature map and equally long name list.
    raise NotImplementedError


def global_pool(feature_map: np.ndarray,
                statistics: tuple[str, ...] = ("mean", "std", "p90")) -> np.ndarray:
    """Pool local channels into one reproducible image descriptor."""
    # YOUR CODE HERE
    #
    # TODO 1 — Flatten spatial positions
    #   - Reshape H x W x D into (H*W) x D without changing channel order.
    #
    # TODO 2 — Implement the supported statistics
    #   - Support mean, std, p10, p50, and p90, each computed per channel.
    #   - Raise ValueError naming an unsupported statistic.
    #
    # TODO 3 — Assemble the descriptor
    #   - Evaluate statistics in the exact caller-supplied order.
    #   - Concatenate their D-vectors and return a one-dimensional float32 array.
    raise NotImplementedError


def feature_family_indices(names: list[str]) -> dict[str, list[int]]:
    """Map colour/Gabor/gradient/edge families to descriptor indices."""
    # YOUR CODE HERE
    #
    # TODO 1 — Create all required keys
    #   - Initialise colour, gabor, gradient, and edge to empty index lists.
    #
    # TODO 2 — Assign channels by their stable name prefix
    #   - Iterate through names in order, take the text before the first "_",
    #     and append the channel index to the matching family.
    #   - Return every family key even when that family has no channels.
    raise NotImplementedError
