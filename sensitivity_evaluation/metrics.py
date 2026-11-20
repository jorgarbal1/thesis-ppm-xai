import numpy as np
import pandas as pd
from scipy.stats import rankdata

from sensitivity_evaluation.sensitivity_perturbations import perturb_numeric, perturb_categorical


def compute_Lp_norm_diff(vec1, vec2, p_norm=2, normalize_to_relative_change=True):
    """L-p norm of vec1-vec2, optionally normalized by vec1's own norm (np.nan if that norm is 0)."""
    vec1, vec2 = vec1.flatten(), vec2.flatten()
    diff = vec1 - vec2
    norm_diff = np.linalg.norm(diff, ord=p_norm)
    if normalize_to_relative_change:
        vec1_norm = np.linalg.norm(vec1, ord=p_norm)
        norm_diff = np.nan if vec1_norm == 0 else norm_diff / vec1_norm
    return norm_diff


def build_representation_vector(x, row_start, length, feature_names, numerical_columns, categorical_columns,
                                 embeddings_by_var):
    """Flattens one prefix into a single vector: numerical columns as-is, categoricals as their embedding."""
    pieces = []
    for name in feature_names:
        if name in numerical_columns:
            col = numerical_columns[name]
            pieces.append(x[row_start:row_start + length, col].astype(np.float64))
        elif name in categorical_columns:
            col = categorical_columns[name]
            embedding = embeddings_by_var[name]
            labels = x[row_start:row_start + length, col].astype(int)
            pieces.append(embedding[labels].reshape(-1).astype(np.float64))
    return np.concatenate(pieces, axis=0)


def relative_stability_montecarlo_v4(
    model, prefix, length, explain_fn, h_fn, feature_names,
    numerical_columns, categorical_columns, columns_info, numerical_minmax,
    gap_columns, implicit_cumulative_columns, percent_cells_numeric, noise_scale, min_noise,
    sampling_distributions_by_var, percent_cells_categorical,
    embeddings_by_var, n_samples, rng, y_minmax, p_norm=2,
):
    """Monte Carlo RIS/ROS over n_samples perturbations of one prefix.

    Returns {"RIS": {"max": float, "mean": float, "median": float, "samples": [float]},
             "ROS": {"max": float, "mean": float, "median": float, "samples": [float]}}."""
    row_start = prefix.shape[0] - length
    y_min, y_max = y_minmax

    def to_days(y_scaled):
        return (float(y_scaled) * (y_max - y_min) + y_min) / 86400

    perturbed_list = []
    for _ in range(n_samples):
        numeric_perturbed, _ = perturb_numeric(
            prefix, length, numerical_columns, columns_info, gap_columns, implicit_cumulative_columns,
            numerical_minmax, percent_cells_numeric, noise_scale, rng, min_noise,
        )
        perturbed, _ = perturb_categorical(
            numeric_perturbed, length, categorical_columns, columns_info, sampling_distributions_by_var,
            percent_cells_categorical, rng,
        )
        perturbed_list.append(perturbed)

    full_batch = np.stack([prefix] + perturbed_list, axis=0)
    e_full = explain_fn(model, full_batch, length, feature_names)
    h_full = h_fn(model, full_batch)

    e_x, e_x_prime_batch = e_full[0], e_full[1:]
    x_repr = build_representation_vector(prefix, row_start, length, feature_names, numerical_columns,
                                         categorical_columns, embeddings_by_var)
    h_x_days = to_days(h_full[0])

    ris_samples, ros_samples = [], []
    for i in range(n_samples):
        xprime_repr = build_representation_vector(perturbed_list[i], row_start, length, feature_names,
                                                    numerical_columns, categorical_columns, embeddings_by_var)
        h_xprime_days = to_days(h_full[1 + i])

        exp_diff = compute_Lp_norm_diff(e_x, e_x_prime_batch[i], p_norm, normalize_to_relative_change=True)

        repr_diff_ris = compute_Lp_norm_diff(x_repr, xprime_repr, p_norm, normalize_to_relative_change=True)
        if not (np.isnan(repr_diff_ris) or np.isnan(exp_diff) or repr_diff_ris == 0):
            ris_samples.append(exp_diff / repr_diff_ris)

        repr_diff_ros = compute_Lp_norm_diff(np.array([h_x_days]), np.array([h_xprime_days]), p_norm,
                                              normalize_to_relative_change=True)
        if not (np.isnan(repr_diff_ros) or np.isnan(exp_diff) or repr_diff_ros == 0):
            ros_samples.append(exp_diff / repr_diff_ros)

    def _summarize(samples):
        if not samples:
            return {"max": np.nan, "mean": np.nan, "median": np.nan, "samples": []}
        return {"max": float(np.max(samples)), "mean": float(np.mean(samples)),
                "median": float(np.median(samples)), "samples": samples}

    return {"RIS": _summarize(ris_samples), "ROS": _summarize(ros_samples)}


def _topk_idx(attribution, k):
    """Top-k feature indices by descending |attribution|."""
    return np.argsort(-np.abs(attribution))[:k]


def _dense_rank(attribution):
    """Dense rank of |attribution| (tied values share a rank)."""
    return rankdata(-np.abs(attribution), method="dense")


def feature_rank_agreement(e_x, e_x_prime, k):
    """Feature Agreement and Rank Agreement between two explanations of the same instance.

    Returns (FA, RA), both floats in [0, 1]."""
    e_x_flat, e_xp_flat = e_x.flatten(), e_x_prime.flatten()
    idx_x, idx_xp = _topk_idx(e_x_flat, k), _topk_idx(e_xp_flat, k)

    set_x, set_xp = set(idx_x.tolist()), set(idx_xp.tolist())
    fa = len(set_x & set_xp) / k

    rank_x, rank_xp = _dense_rank(e_x_flat), _dense_rank(e_xp_flat)
    topk_rank_x, topk_rank_xp = rank_x[idx_x], rank_xp[idx_xp]
    ra = float(np.sum((idx_x == idx_xp) & (topk_rank_x == topk_rank_xp))) / k
    return fa, ra


def relative_stability_montecarlo_v4_with_agreement(
    model, prefix, length, explain_fn, h_fn, feature_names,
    numerical_columns, categorical_columns, columns_info, numerical_minmax,
    gap_columns, implicit_cumulative_columns, percent_cells_numeric, noise_scale, min_noise,
    sampling_distributions_by_var, percent_cells_categorical,
    embeddings_by_var, n_samples, rng, y_minmax, k_agreement, p_norm=2,
):
    """Same Monte Carlo pass as relative_stability_montecarlo_v4, plus Feature/Rank Agreement per perturbation.

    Returns relative_stability_montecarlo_v4's dict, plus:
             "FA": {"min": float, "mean": float, "median": float, "samples": [float]},
             "RA": {"min": float, "mean": float, "median": float, "samples": [float]}}."""
    row_start = prefix.shape[0] - length
    y_min, y_max = y_minmax

    def to_days(y_scaled):
        return (float(y_scaled) * (y_max - y_min) + y_min) / 86400

    perturbed_list = []
    for _ in range(n_samples):
        numeric_perturbed, _ = perturb_numeric(
            prefix, length, numerical_columns, columns_info, gap_columns, implicit_cumulative_columns,
            numerical_minmax, percent_cells_numeric, noise_scale, rng, min_noise,
        )
        perturbed, _ = perturb_categorical(
            numeric_perturbed, length, categorical_columns, columns_info, sampling_distributions_by_var,
            percent_cells_categorical, rng,
        )
        perturbed_list.append(perturbed)

    full_batch = np.stack([prefix] + perturbed_list, axis=0)
    e_full = explain_fn(model, full_batch, length, feature_names)
    h_full = h_fn(model, full_batch)

    e_x, e_x_prime_batch = e_full[0], e_full[1:]
    x_repr = build_representation_vector(prefix, row_start, length, feature_names, numerical_columns,
                                         categorical_columns, embeddings_by_var)
    h_x_days = to_days(h_full[0])

    ris_samples, ros_samples, fa_samples, ra_samples = [], [], [], []
    for i in range(n_samples):
        xprime_repr = build_representation_vector(perturbed_list[i], row_start, length, feature_names,
                                                    numerical_columns, categorical_columns, embeddings_by_var)
        h_xprime_days = to_days(h_full[1 + i])

        exp_diff = compute_Lp_norm_diff(e_x, e_x_prime_batch[i], p_norm, normalize_to_relative_change=True)

        repr_diff_ris = compute_Lp_norm_diff(x_repr, xprime_repr, p_norm, normalize_to_relative_change=True)
        if not (np.isnan(repr_diff_ris) or np.isnan(exp_diff) or repr_diff_ris == 0):
            ris_samples.append(exp_diff / repr_diff_ris)

        repr_diff_ros = compute_Lp_norm_diff(np.array([h_x_days]), np.array([h_xprime_days]), p_norm,
                                              normalize_to_relative_change=True)
        if not (np.isnan(repr_diff_ros) or np.isnan(exp_diff) or repr_diff_ros == 0):
            ros_samples.append(exp_diff / repr_diff_ros)

        fa, ra = feature_rank_agreement(e_x, e_x_prime_batch[i], k_agreement)
        fa_samples.append(fa)
        ra_samples.append(ra)

    def _summarize_max(samples):
        if not samples:
            return {"max": np.nan, "mean": np.nan, "median": np.nan, "samples": []}
        return {"max": float(np.max(samples)), "mean": float(np.mean(samples)),
                "median": float(np.median(samples)), "samples": samples}

    def _summarize_min(samples):
        return {"min": float(np.min(samples)), "mean": float(np.mean(samples)),
                "median": float(np.median(samples)), "samples": samples}

    return {
        "RIS": _summarize_max(ris_samples), "ROS": _summarize_max(ros_samples),
        "FA": _summarize_min(fa_samples), "RA": _summarize_min(ra_samples),
    }
