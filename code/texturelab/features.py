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
    image = image.astype(np.float32)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)  # gray -> fake RGB
    if image.max() > 1.0:
        image = image / 255.0
    gray = rgb2gray(image)
    # TODO 2 — Initialise aligned channel/name groups
    #   - Keep feature chunks and their names in matching, deterministic order.
    #   - If include_colour, append the first three image channels named
    #     colour_r, colour_g, and colour_b.
    chunks = []
    names = []

    if config.include_colour:
        chunks.append(image[..., 0:1])
        names.append("colour_r")
        chunks.append(image[..., 1:2])
        names.append("colour_g")
        chunks.append(image[..., 2:3])
        names.append("colour_b")

    # TODO 3 — Add the Gabor branch when enabled
    #   - Build config.gabor's bank and compute pooled maps using its pool_size
    #     and energy mode.
    #   - Create one stable name from each kernel's frequency/orientation/phase.
    if config.include_gabor:
        bank = make_gabor_bank(config.gabor)
        energy = gabor_energy_maps(gray, bank, config.gabor.pool_size, config.gabor.energy)
        chunks.append(energy)
        for _, meta in bank:
            names.append(f"gabor_f{meta['frequency']:.2f}_o{meta['orientation']:.2f}_p{meta['phase']:.2f}")

    # TODO 4 — Compute edge data only when needed
    #   - Call detect_edges once if gradient or edge channels are requested.
    #   - For gradient energy, square magnitude, box-pool it with density_size,
    #     add a singleton channel axis, and name it gradient_energy.
    
    if config.include_gradient or config.include_edges:
        edge = detect_edges(gray, config.edge)
    if config.include_gradient:
        grad_energy = edge["magnitude"] ** 2
        pooled_grad = box_mean(grad_energy, config.edge.density_size)
        chunks.append(pooled_grad[..., np.newaxis])
        names.append("gradient_energy")
    # TODO 5 — Add total and orientation-specific edge density
    #   - Pool the Boolean edge map for edge_density.
    #   - Map directions modulo pi, divide the unsigned orientation range into
    #     config.edge.n_orientations bins, and pool edges belonging to each bin.
    #   - Name bins edge_orientation_0, edge_orientation_1, and so on.
    if config.include_edges:
        edge_density = box_mean(edge["edges"].astype(np.float32), config.edge.density_size)
        chunks.append(edge_density[..., np.newaxis])
        names.append("edge_density")
    directions_mod = edge["direction"] % np.pi
    n_bins = config.edge.n_orientations
    bin_width = np.pi / n_bins
    for i in range(n_bins):
        low_angle = i * bin_width
        high_angle = (i + 1) * bin_width
        in_bin = (directions_mod >= low_angle) & (directions_mod < high_angle)
        masked = edge["edges"] & in_bin
        pooled_bin = box_mean(masked.astype(np.float32), config.edge.density_size)
        chunks.append(pooled_bin[..., np.newaxis])
        names.append(f"edge_orientation_{i}")

    # TODO 6 — Assemble and optionally standardise
    #   - Reject a configuration with no enabled feature family.
    #   - Concatenate chunks along the final axis and return float32.
    #   - If standardise_per_image, standardise every channel over H/W using a
    #     small epsilon; return the feature map and equally long name list.
    if not chunks:
        raise ValueError("no enabled feature family")
    features = np.concatenate(chunks, axis=-1).astype(np.float32)
    if config.standardise_per_image:
        mean = features.mean(axis=(0, 1), keepdims=True)
        std = features.std(axis=(0, 1), keepdims=True)
        eps = 1e-6
        features = (features - mean) / (std + eps)

    return features, names


def global_pool(feature_map: np.ndarray,
                statistics: tuple[str, ...] = ("mean", "std", "p90")) -> np.ndarray:
    """Pool local channels into one reproducible image descriptor."""
    # YOUR CODE HERE
    #
    # TODO 1 — Flatten spatial positions
    #   - Reshape H x W x D into (H*W) x D without changing channel order.
    H, W, D = feature_map.shape
    flat = feature_map.reshape(H * W, D)
    # TODO 2 — Implement the supported statistics
    #   - Support mean, std, p10, p50, and p90, each computed per channel.
    #   - Raise ValueError naming an unsupported statistic.
    results = []
    for stat in statistics:
        if stat == "mean":
            results.append(flat.mean(axis=0))
        elif stat == "std":
            results.append(flat.std(axis=0))
        elif stat == "p10":
            results.append(np.percentile(flat, 10, axis=0))
        elif stat == "p50":
            results.append(np.percentile(flat, 50, axis=0))
        elif stat == "p90":
            results.append(np.percentile(flat, 90, axis=0))
        else:
            raise ValueError(f"unsupported statistic: {stat}")
    # TODO 3 — Assemble the descriptor
    #   - Evaluate statistics in the exact caller-supplied order.
    #   - Concatenate their D-vectors and return a one-dimensional float32 array.
    return np.concatenate(results).astype(np.float32)


def feature_family_indices(names: list[str]) -> dict[str, list[int]]:
    """Map colour/Gabor/gradient/edge families to descriptor indices."""
    # YOUR CODE HERE
    #
    # TODO 1 — Create all required keys
    #   - Initialise colour, gabor, gradient, and edge to empty index lists.
    families = {"colour": [], "gabor": [], "gradient": [], "edge": []}
    # TODO 2 — Assign channels by their stable name prefix
    #   - Iterate through names in order, take the text before the first "_",
    #     and append the channel index to the matching family.
    #   - Return every family key even when that family has no channels.
    for i, name in enumerate(names):
        prefix = name.split("_")[0]
        if prefix in families:
            families[prefix].append(i)

    return families
