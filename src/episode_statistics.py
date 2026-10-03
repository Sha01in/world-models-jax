"""Percentile bootstrap intervals that keep correlated episode frames together."""

import numpy as np


def episode_bootstrap(episode_ids, statistic, *, resamples=5000, seed=73501):
    """Evaluate a vector statistic on resampled whole episodes.

    ``statistic(indices)`` must use the same row indices for all paired arrays.
    The fitted model is fixed: these intervals capture held-out episode sampling
    uncertainty, not uncertainty from fitting the model or changing color labels.
    """
    episode_ids = np.asarray(episode_ids)
    if episode_ids.ndim != 1 or not len(episode_ids) or resamples < 1:
        raise ValueError("Nonempty episode IDs and positive resample count required")
    groups = [
        np.flatnonzero(episode_ids == identity) for identity in np.unique(episode_ids)
    ]
    point = np.atleast_1d(
        np.asarray(statistic(np.arange(len(episode_ids))), np.float64)
    )
    if point.ndim != 1:
        raise ValueError("Statistic must return a scalar or one-dimensional vector")
    rng = np.random.default_rng(seed)
    samples = np.empty((resamples, len(point)), np.float64)
    for index in range(resamples):
        choices = rng.integers(len(groups), size=len(groups))
        rows = np.concatenate([groups[choice] for choice in choices])
        value = np.atleast_1d(np.asarray(statistic(rows), np.float64))
        if value.shape != point.shape:
            raise ValueError("Bootstrap statistic changed shape")
        samples[index] = value
    intervals, valid_counts = [], []
    for column in samples.T:
        finite = column[np.isfinite(column)]
        valid_counts.append(len(finite))
        intervals.append(
            np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else None
        )
    return dict(
        unit="whole held-out episodes; frames retained together",
        episodes=len(groups),
        frames=len(episode_ids),
        resamples=resamples,
        seed=seed,
        confidence_level=0.95,
        method="percentile bootstrap, fitted probe held fixed",
        point_estimates=[float(v) if np.isfinite(v) else None for v in point],
        intervals=intervals,
        valid_resamples=valid_counts,
    )
