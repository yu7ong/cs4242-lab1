"""Per-material diagonal normal model and dense anomaly prediction."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .config import NormalityConfig
from .supplied import box_mean


@dataclass
class NormalModel:
    # Store the fitted normal model statistics and threshold. [AI-Code] [Human-Check]
    mean: np.ndarray
    std: np.ndarray
    threshold: float
    feature_names: list[str] | None = None


def _scores(
    features: np.ndarray, mean: np.ndarray, std: np.ndarray, epsilon: float
) -> np.ndarray:
    return np.sqrt(np.mean(((features - mean) / (std + epsilon)) ** 2, axis=-1)).astype(
        np.float32
    )


def _pool_score(score: np.ndarray, config: NormalityConfig) -> np.ndarray:
    """Suppress isolated patch responses while preserving coherent anomalies. [AI-Code] [Human-Check]"""
    if config.score_pool_size < 1 or config.score_pool_size % 2 == 0:
        raise ValueError("score_pool_size must be a positive odd integer")
    return box_mean(score, config.score_pool_size).astype(np.float32)


def _interior(score: np.ndarray, border: int) -> np.ndarray:
    if border <= 0:
        return score
    if 2 * border >= min(score.shape):
        raise ValueError("ignore_border is too large for the score map")
    return score[border:-border, border:-border]


def fit_normal_model(feature_maps: list[np.ndarray] | np.ndarray, config: NormalityConfig,
                     feature_names: list[str] | None = None) -> NormalModel:
    """Fit normal feature statistics using normal training maps only. [AI-Code] [Human-Check]"""
    
    maps = [feature_maps] if isinstance(feature_maps, np.ndarray) else list(feature_maps)
    if not maps:
        raise ValueError("feature_maps must be a non-empty collection")

    n_features = maps[0].shape[-1]
    if any(m.shape[-1] != n_features for m in maps):
        raise ValueError("all feature maps must share the same feature dimension")

    samples = np.concatenate([m.reshape(-1, n_features) for m in maps], axis=0)
    if samples.shape[0] > config.max_samples:
        rng = np.random.default_rng(config.random_seed)
        indices = rng.choice(samples.shape[0], size=config.max_samples, replace=False)
        samples = samples[indices]


    mean = samples.mean(axis=0)
    std = samples.std(axis=0)


    if config.mask_threshold is not None:
        threshold = config.mask_threshold
    else:
        pooled_scores = [
            _interior(_pool_score(_scores(m, mean, std, config.epsilon), config), config.ignore_border)
            for m in maps
        ]
        normal_scores = np.concatenate([s.reshape(-1) for s in pooled_scores])
        threshold = np.percentile(normal_scores, config.threshold_percentile)


    return NormalModel(
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        threshold=float(threshold),
        feature_names=feature_names,
    )


def predict_anomaly(feature_map: np.ndarray, model: NormalModel, config: NormalityConfig
                    ) -> tuple[np.ndarray, float, np.ndarray]:
    """Return dense score, image-level score, and predicted mask. [AI-Code] [Human-Check]"""

    if feature_map.shape[-1] != model.mean.shape[-1]:
        raise ValueError("feature_map's feature dimension does not match the fitted model")

    raw_score = _scores(feature_map, model.mean, model.std, config.epsilon)
    score = _pool_score(raw_score, config)


    interior_score = _interior(score, config.ignore_border)
    image_score = np.percentile(interior_score, config.image_percentile)

    mask = score >= model.threshold
    border = config.ignore_border
    if border > 0:
        mask[:border, :] = False
        mask[-border:, :] = False
        mask[:, :border] = False
        mask[:, -border:] = False

    return score.astype(np.float32), float(image_score), mask.astype(bool)


def select_mask_threshold(score_maps: list[np.ndarray], masks: list[np.ndarray],
                          candidates: int = 80) -> float:
    """Choose the pixel-F1-optimal threshold on public validation only. [AI-Code] [Human-Check]"""

    if not score_maps or not masks:
        raise ValueError("score_maps and masks must be non-empty")
    if len(score_maps) != len(masks):
        raise ValueError("score_maps and masks must have matching lengths")
    if candidates < 2:
        raise ValueError("candidates must be >= 2")

    scores = []
    bool_masks = []
    for s, m in zip(score_maps, masks):
        s = np.asarray(s, dtype=float)
        m = np.asarray(m, dtype=bool)
        if s.shape != m.shape:
            raise ValueError("each score map and mask must share the same shape")
        if not np.all(np.isfinite(s)):
            raise ValueError("score maps must contain only finite values")
        scores.append(s)
        bool_masks.append(m)


    flat_scores = np.concatenate([s.reshape(-1) for s in scores])
    flat_masks = np.concatenate([m.reshape(-1) for m in bool_masks])
    quantile_levels = np.linspace(0.5, 0.999, candidates)
    thresholds = np.quantile(flat_scores, quantile_levels)


    best_f1 = -1.0
    best_threshold = float(thresholds[0])
    for threshold in thresholds:
        predicted = flat_scores >= threshold
        tp = np.count_nonzero(predicted & flat_masks)
        fp = np.count_nonzero(predicted & ~flat_masks)
        fn = np.count_nonzero(~predicted & flat_masks)
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold
