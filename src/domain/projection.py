import numpy as np
from sklearn.manifold import TSNE


_TSNE_MIN_PERPLEXITY = 5       # below this t-SNE degenerates to noise
_TSNE_MAX_PERPLEXITY = 30      # library default upper bound

def stratified_subsample(
    labels: np.ndarray,
    *,
    n_samples: int | None = None,
    stratify: bool = True,
    noise_mask: np.ndarray | None = None,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Sample up to n_samples indices from labels, stratified by group.

    Returns an indices array. If n_samples is None or >= len(labels), returns all indices.
    When stratify=True, samples proportionally to group size (at least one per group).
    When stratify=False, samples equally per group, capped at the smallest group.
    If noise_mask is provided, noise points fill the remaining budget after stratified sampling.
    """
    n = len(labels)
    if n_samples is None or n_samples >= n:
        return np.arange(n)

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    noise = (
        np.asarray(noise_mask, dtype=bool)
        if noise_mask is not None
        else np.zeros(n, dtype=bool)
    )
    valid_idx = np.where(~noise)[0]
    valid_labels = labels[valid_idx]
    unique_groups = np.unique(valid_labels)
    counts = np.array([np.sum(valid_labels == g) for g in unique_groups])
    budget_valid = min(n_samples, len(valid_idx))

    if stratify:
        per_group = np.maximum(
            1, np.round(counts / counts.sum() * budget_valid).astype(int)
        )
    else:
        min_count = int(counts.min())
        per_group = np.full(
            len(unique_groups),
            min(budget_valid // len(unique_groups), min_count),
            dtype=int,
        )

    parts = []
    for g, take in zip(unique_groups, per_group):
        pool = valid_idx[valid_labels == g]
        parts.append(rng.choice(pool, min(take, len(pool)), replace=False))

    if noise_mask is not None:
        used = sum(len(p) for p in parts)
        noise_idx = np.where(noise)[0]
        remaining = max(0, n_samples - used)
        if remaining and len(noise_idx):
            parts.append(
                rng.choice(noise_idx, min(remaining, len(noise_idx)), replace=False)
            )

    return np.concatenate(parts) if parts else np.array([], dtype=int)


def tsne_projection(
    data: np.ndarray,
    *,
    n_components: int = 2,
    perplexity: int | None = None,
    random_state: int = 42,
) -> np.ndarray:
    """Project data to lower dimensions using t-SNE with adaptive perplexity."""
    if perplexity is None:
        perplexity = max(
            _TSNE_MIN_PERPLEXITY,
            min(_TSNE_MAX_PERPLEXITY, (len(data) - 1) // 3),
        )
    return TSNE(
        n_components=n_components, perplexity=perplexity, random_state=random_state
    ).fit_transform(data)
