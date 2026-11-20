import os
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from preprocessing.loader import load_next_activity, load_remaining_time

LOADERS = {
    "next_activity": load_next_activity,
    "remaining_time": load_remaining_time,
}


def _processed_path(dataset_name, processed_root):
    return os.path.join(processed_root, dataset_name, "processed_multi")


def load_dataset(task, dataset_name, processed_root):
    """Loads a processed dataset's arrays via preprocessing.loader, for the given task/dataset/root."""
    if task not in LOADERS:
        raise ValueError(f"unknown task {task!r}, expected one of {list(LOADERS)}")
    loader_fn = LOADERS[task]
    return loader_fn(_processed_path(dataset_name, processed_root))


def perturb_numeric(prefix, length, numerical_columns, columns_info, gap_columns, implicit_cumulative_columns,
                     numerical_minmax, percent_cells, noise_scale, rng, min_noise = 1.0):
    """One Gaussian-noise perturbation of the numerical cells of a single left-padded prefix."""
    if not (0.0 <= percent_cells <= 1.0):
        raise ValueError(f"percent_cells must be in [0, 1], got {percent_cells}")
    if noise_scale < 0:
        raise ValueError(f"noise_scale must be >= 0, got {noise_scale}")
    if length < 1 or length > prefix.shape[0]:
        raise ValueError(f"length ({length}) must be in [1, prefix length ({prefix.shape[0]})]")

    dependents = {dep for deps in gap_columns.values() for dep, _ in deps}
    missing = [name for name in gap_columns if name not in numerical_columns] + \
              [dep for dep in dependents if dep not in numerical_columns] + \
              [name for name in implicit_cumulative_columns if name not in numerical_columns]
    if missing:
        raise ValueError(f"gap_columns/implicit_cumulative_columns refer to unknown numerical variable(s): {missing}")

    perturbed = prefix.copy()
    row_start = perturbed.shape[0] - length

    event_vars = [name for name in numerical_columns if name not in dependents and columns_info[name] == "event"]
    case_vars = [name for name in numerical_columns if name not in dependents and columns_info[name] == "case"]

    def event_range(name):
        return range(1, length) if name in gap_columns else range(length)

    candidates = [(name, t) for name in event_vars for t in event_range(name)] + \
                 [(name, None) for name in case_vars]
    if len(candidates) == 0:
        n_touch = 0
    elif percent_cells <= 0:
        n_touch = 0
    else:
        n_touch = min(max(round(percent_cells * len(candidates)), 1), len(candidates))
    if percent_cells > 0 and len(candidates) == 0:
        warnings.warn("perturb_numeric: no eligible numerical cell to perturb for this prefix "
                      "(length={length}, no event/case numerical columns) -- returning an unchanged prefix."
                      .format(length=length))
    chosen = [candidates[i] for i in rng.choice(len(candidates), size = n_touch, replace = False)] if n_touch else []

    by_implicit = {}
    touched = []
    for name, t in chosen:
        if name in implicit_cumulative_columns:
            by_implicit.setdefault(name, set()).add(t)
            touched.append((name, t))
            continue

        col = numerical_columns[name]
        vmin, vmax = numerical_minmax[name]
        span = vmax - vmin if vmax > vmin else 1.0

        if t is None:
            raw_value = perturbed[row_start, col] * span + vmin
            noisy = raw_value + rng.normal(0.0, noise_scale * max(abs(raw_value), min_noise))
            perturbed[row_start:row_start + length, col] = (noisy - vmin) / span
        else:
            r = row_start + t
            raw_value = perturbed[r, col] * span + vmin
            noisy = raw_value + rng.normal(0.0, noise_scale * max(abs(raw_value), min_noise))
            if name in gap_columns:
                noisy = max(noisy, 0.0)
            perturbed[r, col] = (noisy - vmin) / span
        touched.append((name, t))

    for gap_name, deps in gap_columns.items():
        gap_col = numerical_columns[gap_name]
        gap_vmin, gap_vmax = numerical_minmax[gap_name]
        gap_span = gap_vmax - gap_vmin if gap_vmax > gap_vmin else 1.0

        orig_gap = prefix[row_start:row_start + length, gap_col] * gap_span + gap_vmin
        new_gap = perturbed[row_start:row_start + length, gap_col] * gap_span + gap_vmin
        shift = np.cumsum(new_gap) - np.cumsum(orig_gap)

        for dep_name, modulus in deps:
            dep_col = numerical_columns[dep_name]
            dep_vmin, dep_vmax = numerical_minmax[dep_name]
            dep_span = dep_vmax - dep_vmin if dep_vmax > dep_vmin else 1.0

            if modulus is None:
                new_dep = np.cumsum(new_gap)
            else:
                orig_dep = prefix[row_start:row_start + length, dep_col] * dep_span + dep_vmin
                new_dep = (orig_dep + shift) % modulus
            perturbed[row_start:row_start + length, dep_col] = (new_dep - dep_vmin) / dep_span

    for name, touched_t in by_implicit.items():
        col = numerical_columns[name]
        modulus = implicit_cumulative_columns[name]
        vmin, vmax = numerical_minmax[name]
        span = vmax - vmin if vmax > vmin else 1.0

        raw_col = perturbed[row_start:row_start + length, col] * span + vmin
        gap = np.diff(raw_col, prepend = 0.0)
        for t in touched_t:
            gap[t] = gap[t] + rng.normal(0.0, noise_scale * max(abs(gap[t]), min_noise))
            if t > 0:
                gap[t] = max(gap[t], 0.0)
        new_raw = np.cumsum(gap)
        if modulus is not None:
            new_raw = new_raw % modulus
        perturbed[row_start:row_start + length, col] = (new_raw - vmin) / span

    return perturbed, touched


def clean_prefix_mask(X, k_array, gap_col):
    """Mask of rows whose gap column has no zero within the real prefix."""
    max_len = X.shape[1]
    mask = np.ones(len(X), dtype = bool)
    for i in range(len(X)):
        length = int(k_array[i]) + 1
        row_start = max_len - length
        gaps = X[i, row_start + 1:row_start + length, gap_col]
        mask[i] = not np.any(gaps == 0.0)
    return mask


def extract_embedding(model, variable_name):
    """Returns the (vocab_size, embed_dim) embedding table of a categorical variable's layer."""
    import re
    layer_name = f"cat_embed_{re.sub(r'[^A-Za-z0-9_.-]', '_', variable_name)}"
    return model.get_layer(layer_name).embedding.get_weights()[0]


def build_event_level_transitions(X_train, train_k, column):
    """Counts label transitions and start labels of an event-level categorical column over X_train."""
    transitions, starts = {}, Counter()
    max_len = X_train.shape[1]
    for i in range(len(X_train)):
        length = int(train_k[i]) + 1
        row_start = max_len - length
        labels = X_train[i, row_start:row_start + length, column].astype(int)
        starts[int(labels[0])] += 1
        for prev, nxt in zip(labels[:-1], labels[1:]):
            transitions.setdefault(int(prev), Counter())[int(nxt)] += 1
    return transitions, starts


def _rrf_distribution(candidates, rankings, k_rrf):
    """Reciprocal-rank-fusion sampling distribution over candidates, from several per-candidate rankings."""
    scores = np.array([sum(1.0 / (k_rrf + ranking[c]) for ranking in rankings) for c in candidates])
    return candidates, scores / scores.sum()


def _rank_of(counts_or_candidates, better_first):
    """Rank (1 = best) of each candidate under better_first's ordering."""
    order = sorted(counts_or_candidates, key = better_first)
    return {label: i + 1 for i, label in enumerate(order)}


def build_transition_sampling_distributions(transitions, starts, embedding, k_rrf = 10):
    """Per-(previous label) RRF sampling distribution, fusing transition frequency and embedding distance."""
    all_labels = set(starts) | set(transitions) | {l for counts in transitions.values() for l in counts}
    distributions = {}

    def add(previous_key, pool):
        for original in all_labels:
            candidates = [label for label in pool if label != original]
            if not candidates:
                distributions[(previous_key, original)] = None
                continue
            freq_rank = _rank_of(candidates, better_first = lambda l: -pool[l])
            emb_rank = _rank_of(candidates, better_first = lambda l: np.linalg.norm(embedding[l] - embedding[original]))
            distributions[(previous_key, original)] = _rrf_distribution(candidates, [freq_rank, emb_rank], k_rrf)

    add(None, starts)
    for previous_label, pool in transitions.items():
        add(previous_label, pool)
    return distributions


def build_embedding_only_sampling_distributions(embedding, valid_labels, k_rrf = 10):
    """Per-label sampling distribution based on embedding distance alone (no transition structure)."""
    distributions = {}
    for original in valid_labels:
        candidates = [label for label in valid_labels if label != original]
        if not candidates:
            distributions[original] = None
            continue
        emb_rank = _rank_of(candidates, better_first = lambda l: np.linalg.norm(embedding[l] - embedding[original]))
        distributions[original] = _rrf_distribution(candidates, [emb_rank], k_rrf)
    return distributions


def build_all_categorical_sampling_distributions(X_train, train_k, categorical_columns, columns_info,
                                                   embeddings_by_var, k_rrf = 10):
    """Builds a sampling distribution per categorical variable: transition-based for event-level, embedding-only for case-level."""
    distributions_by_var = {}
    for name, col in categorical_columns.items():
        embedding = embeddings_by_var[name]
        if columns_info[name] == "event":
            transitions, starts = build_event_level_transitions(X_train, train_k, col)
            distributions_by_var[name] = build_transition_sampling_distributions(transitions, starts, embedding, k_rrf)
        else:
            vocab_size = embedding.shape[0]
            valid_labels = range(1, vocab_size - 1)
            distributions_by_var[name] = build_embedding_only_sampling_distributions(embedding, valid_labels, k_rrf)
    return distributions_by_var


def perturb_categorical(prefix, length, categorical_columns, columns_info, sampling_distributions_by_var, percent_cells, rng):
    """One sampled-replacement perturbation of the categorical cells of a single left-padded prefix."""
    if not (0.0 <= percent_cells <= 1.0):
        raise ValueError(f"percent_cells must be in [0, 1], got {percent_cells}")
    if length < 1 or length > prefix.shape[0]:
        raise ValueError(f"length ({length}) must be in [1, prefix length ({prefix.shape[0]})]")

    perturbed = prefix.copy()
    row_start = perturbed.shape[0] - length
    names = [name for name in categorical_columns if name in sampling_distributions_by_var]
    missing_vars = [name for name in categorical_columns if name not in sampling_distributions_by_var]
    if missing_vars:
        warnings.warn(f"perturb_categorical: {missing_vars} have no sampling distribution and will never "
                      "be perturbed -- pass them through build_all_categorical_sampling_distributions first "
                      "if they should be eligible.")
    event_names = [name for name in names if columns_info[name] == "event"]
    case_names = [name for name in names if columns_info[name] == "case"]

    candidates = [(name, t) for name in event_names for t in range(length)] + [(name, None) for name in case_names]
    if len(candidates) == 0:
        n_touch = 0
    elif percent_cells <= 0:
        n_touch = 0
    else:
        n_touch = min(max(round(percent_cells * len(candidates)), 1), len(candidates))
    if percent_cells > 0 and len(candidates) == 0:
        warnings.warn(f"perturb_categorical: no eligible categorical cell to perturb for this prefix "
                      f"(length={length}) -- returning an unchanged prefix.")

    order = rng.permutation(len(candidates)) if candidates else np.array([], dtype=int)
    touched = []
    for i in order:
        if len(touched) >= n_touch:
            break
        name, t = candidates[i]
        col = categorical_columns[name]
        distributions = sampling_distributions_by_var[name]

        if t is None:
            original = int(perturbed[row_start, col])
            entry = distributions.get(original)
            if entry is None:
                continue
            new_label = rng.choice(entry[0], p = entry[1])
            perturbed[row_start:row_start + length, col] = new_label
        else:
            r = row_start + t
            original = int(perturbed[r, col])
            previous_label = None if t == 0 else int(prefix[r - 1, col])
            entry = distributions.get((previous_label, original))
            if entry is None:
                continue
            new_label = rng.choice(entry[0], p = entry[1])
            perturbed[r, col] = new_label
        touched.append((name, t))

    if n_touch > 0 and not touched:
        warnings.warn(f"perturb_categorical: all {len(candidates)} candidate cell(s) had a degenerate "
                      "sampling distribution (no alternative label available) -- returning an unchanged prefix.")

    return perturbed, touched


def prefix_length_counts(data):
    """Prefix-length value counts for train and test."""
    return {
        "train": pd.Series(data["train_k"] + 1).value_counts().sort_index(),
        "test": pd.Series(data["test_k"] + 1).value_counts().sort_index(),
    }
