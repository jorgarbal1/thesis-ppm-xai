"""Fourth iteration of the stability metrics: maximizes direct reuse of OpenXAI's own published code
(Agarwal, Saxena, Krishna, Pawelczyk, Johnson, Puri, Zitnik, Lakkaraju, "OpenXAI: Towards a Transparent
Evaluation of Model Explanations", NeurIPS 2022 Datasets & Benchmarks Track, arXiv:2203.06877 2206.11104
-- repo: https://github.com/AI4LIFE-GROUP/OpenXAI, files openxai/experiment_utils.py and openxai/metrics.py,
fetched verbatim on 2026-11-11), instead of our own independently-derived ratio-of-norms formula
(metrics3.py). Every function/block copied essentially as-is from that repo is marked "Directly from
OpenXAI" in its docstring; every point where we deviate (because OpenXAI's code assumes tabular,
fixed-length, classification inputs, and ours are variable-length process-sequence prefixes with a
regression target) is marked "MODIFIED" inline, with a one-line reason.

What's reused as-is from OpenXAI:
  - compute_Lp_norm_diff: their exact normalization utility (norm of the difference, divided by the norm
    of the first vector, AFTER norming -- not cell-by-cell -- with np.nan, not an epsilon floor, as the
    only zero-denominator guard).
  - _single_idx_stability's core loop logic: for each perturbation, compute repr_diff/exp_diff via
    compute_Lp_norm_diff; if either is NaN or repr_diff == 0 (the perturbation didn't actually change
    anything, or the representation itself is degenerate), DISCARD that perturbation entirely (not: give
    it an inflated value). This is the fix for exactly the EG/idx=3174 catastrophic-outlier case found in
    the Sensitivity_Study_3.ipynb investigation: metrics3.py's eps_min floor turned a perturbation that
    changed nothing into a 97621-magnitude outlier; OpenXAI's own code would have just dropped that one
    sample.
  - _single_k_gt_faith's 'overlap'/'rank' branches: the exact Feature Agreement / Rank Agreement
    set-and-rank-overlap computation, including its dense-rank tie-handling.
  - build_representation_vector + RIS's x-side: builds a representation of x, builds one of x', and lets
    compute_Lp_norm_diff diff them -- the exact literal OpenXAI pattern, used for every one of RIS/ROS's
    three ratios (explanation, input, output), not just two of three. An earlier version of this file
    instead special-cased RIS's x-side with its own pre-diffed Diff/Mag matrices, reasoning that collapsing
    each categorical cell to one distance scalar (||v-v'||) was numerically necessary to stop a
    wide-embedding categorical from "outweighing" a numerical column in the Frobenius norm. That reasoning
    was caught and PROVEN WRONG in the conversation this file was built in: by the Euclidean norm's
    block-decomposition identity, ||[a,b]||_2^2 = ||a||_2^2 + ||b||_2^2 for ANY partition of a vector into
    blocks, so collapsing before combining gives EXACTLY the same number (verified numerically to zero
    difference) as keeping every raw embedding component in one big flattened vector and taking a single
    L2 norm over everything. That non-weighting property is specific to the ratio-of-norms formulation
    used here (division happens once, at the very end) -- it does NOT hold for metrics2.py's original
    elementwise-relativize-then-norm convention, where each embedding component is divided by its own
    magnitude individually BEFORE the sum, a per-component nonlinearity that genuinely does not commute
    with collapsing. Since the two give identical numbers here, the simpler, more literally-OpenXAI code
    path (build_representation_vector + compute_Lp_norm_diff, no separate Diff/Mag machinery) is kept.

What's necessarily built from scratch here (OpenXAI's code has no equivalent, because tabular
fixed-length classification inputs don't need it):
  - build_representation_vector: the one piece that must differ from pure OpenXAI, and the one genuinely
    unavoidable reason -- not dimensionality weighting (see above), but that the RAW label-encoded integer
    a categorical column stores (e.g. "category 5" vs "category 3") carries no meaningful numeric distance
    at all: adjacent codes can be semantically opposite, far-apart codes can behave near-identically. Using
    each category's own trained embedding vector (all of it, concatenated in, no per-cell collapse) instead
    of its raw code is the fix -- everything else about this function (numerical columns: the model's own
    already-scaled [0,1]-ish value) mirrors how OpenXAI's own `input`/`x_prime` are already flat, directly
    comparable vectors.
  - The whole Monte Carlo perturbation generation (perturb_numeric/perturb_categorical, imported from
    sensitivity_perturbations2.py, not OpenXAI -- OpenXAI has its own NormalPerturbation/
    get_perturbed_inputs machinery, built for i.i.d. tabular feature perturbation, which does not carry
    over to the gap->cumsum cascade / RRF-sampled categorical perturbation this project already validated).
  - ROS's reference quantity uses h(x) converted to days (MODIFIED: OpenXAI's ROS uses
    model.predict_with_logits(x) directly, appropriate for a classification logit that is already in a
    self-calibrated, comparable scale; our target is a regression value whose raw min-max-scaled range is
    an arbitrary preprocessing artifact -- see metrics3.py's module docstring and sensitivity_formulas.tex
    for why. The REST of ROS -- divide by the norm of this reference quantity itself, not by anything
    else -- is unmodified OpenXAI.)
  - max/mean/median instead of max alone (MODIFIED: OpenXAI's eval_relative_stability only ever computes
    the max per instance, then averages that max across instances; we additionally keep mean/median across
    the N perturbations of ONE instance, since the mean/max alone can be dominated by a single
    pathological perturbation sample -- see the EG outlier investigation this project already did).
  - Feature Agreement / Rank Agreement compare e_x against each perturbed e_x' (MODIFIED: OpenXAI's
    _single_k_gt_faith compares an explanation against a synthetic ground-truth explanation, for a
    faithfulness question; we have no ground truth for a real trained model, so we repurpose the same
    set/rank-overlap formula for a stability question instead -- comparing the original explanation
    against each perturbed one, with the same Monte Carlo protocol as RIS/ROS). min (not max) is reported
    as the "worst case" aggregate, since Feature/Rank Agreement are similarity metrics (higher = more
    stable), the mirror image of RIS/ROS's max convention.

STILL OPEN (flagged in the conversation this file was built in, deliberately not resolved here, to discuss
with the tutor alongside the current "simple" approach below): even though the categorical-embedding
collapse doesn't change the RIS result (see above), a DIFFERENT, real concern remains unaddressed -- a
category's embedding has no constrained scale (its norm depends on how it happened to be trained), while a
numerical column is always in [0,1]. A categorical variable can therefore contribute a systematically
larger or smaller "typical" magnitude than a numerical one, for reasons that have nothing to do with how
much it actually matters. Two candidate fixes discussed but NOT implemented (to test and bring both to the
tutor): (a) normalize each category's embedding to unit norm at lookup time (v/||v||), so every categorical
variable's typical magnitude is bounded the same way a scaled numerical one is; (b) anchor each categorical
variable's distances to its own empirical typical pairwise embedding distance in train (same spirit as the
std(y_train_days) idea for ROS's cross-dataset comparability). Neither is implemented in this file -- the
current code uses raw (non-unit-normalized) embeddings, exactly as extract_embedding returns them.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from sensitivity_evaluation.sensitivity_perturbations import perturb_numeric, perturb_categorical


# ==== Directly from OpenXAI (openxai/experiment_utils.py, compute_Lp_norm_diff) ==== #

def compute_Lp_norm_diff(vec1, vec2, p_norm=2, normalize_to_relative_change=True):
    """
    Computes the L-p norm difference between two vectors.
    Directly from OpenXAI (openxai/experiment_utils.py's compute_Lp_norm_diff), verbatim except for
    dropping the torch/np.array conversion step (our vec1/vec2 are always already plain numpy arrays).
    :param vec1: np.array
    :param vec2: np.array
    :param p_norm: int, float
    :param normalize_to_relative_change: bool
    :return: float (np.nan if normalize_to_relative_change and vec1's own norm is exactly 0)
    """
    vec1, vec2 = vec1.flatten(), vec2.flatten()
    diff = vec1 - vec2
    norm_diff = np.linalg.norm(diff, ord=p_norm)
    if normalize_to_relative_change:
        vec1_norm = np.linalg.norm(vec1, ord=p_norm)
        norm_diff = np.nan if vec1_norm == 0 else norm_diff / vec1_norm
    return norm_diff


# ==== MODIFIED: x-side representation for RIS (no OpenXAI equivalent, see module docstring) ==== #

def build_representation_vector(x, row_start, length, feature_names, numerical_columns, categorical_columns,
                                 embeddings_by_var):
    """Flattens one prefix (or perturbed copy) into the single flat vector compute_Lp_norm_diff expects --
    the literal OpenXAI pattern (build a representation of x, build one of x', let compute_Lp_norm_diff
    diff them), used here for RIS's x-side too (see module docstring for why the categorical-embedding
    collapse this file used to do instead turned out to be numerically identical, and was dropped in
    favour of this simpler, more-literally-OpenXAI path).

    Numerical columns: the already-scaled [0,1]-ish value, one scalar per (event, variable) -- same
    reasoning as metrics3.py (this is the actual input space the model/explainer sees, not raw units).
    Categorical columns: the category's full trained embedding vector (all embed_dim components,
    concatenated in -- not collapsed to a single distance scalar; see module docstring for why that
    collapse is numerically inert here, so this simpler form is preferred).

    x: raw (max_len, F_raw) array (prefix or perturbed copy), row_start/length: real-event window.
    feature_names, numerical_columns, categorical_columns, embeddings_by_var: same convention as
    metrics2.py/metrics3.py's build_input_change_matrix / build_diff_and_magnitude_matrices.

    Returns a 1-D numpy float64 array (flattened across events and variables)."""
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


# ==== Directly from OpenXAI (openxai/metrics.py, _single_idx_stability's core loop), MODIFIED where noted ==== #

def relative_stability_montecarlo_v4(
    model, prefix, length, explain_fn, h_fn, feature_names,
    numerical_columns, categorical_columns, columns_info, numerical_minmax,
    gap_columns, implicit_cumulative_columns, percent_cells_numeric, noise_scale, min_noise,
    sampling_distributions_by_var, percent_cells_categorical,
    embeddings_by_var, n_samples, rng, y_minmax, p_norm=2,
):
    """Monte Carlo RIS/ROS over n_samples perturbations of ONE prefix, following OpenXAI's
    _single_idx_stability loop logic as closely as this domain allows (see module docstring for exactly
    what is copied vs. necessarily rebuilt).

    MODIFIED vs. OpenXAI: keeps every valid (non-discarded) stability_measure in a list, instead of only
    tracking a running max -- so max/mean/median can all be reported, not just max (see module docstring).
    A perturbation is discarded (never added to the list) exactly when OpenXAI's code would have `continue`d:
    repr_diff is NaN, exp_diff is NaN, or repr_diff == 0 (the perturbation changed nothing, or the
    reference quantity's own norm is exactly 0) -- there is no eps_min floor anywhere in this file.

    model, prefix, length, explain_fn, h_fn, feature_names, numerical_columns, categorical_columns,
    columns_info, numerical_minmax, gap_columns, implicit_cumulative_columns, percent_cells_numeric,
    noise_scale, min_noise, sampling_distributions_by_var, percent_cells_categorical, embeddings_by_var,
    n_samples, rng: identical meaning to metrics3.py's relative_stability_montecarlo_v3.
    y_minmax: (y_min, y_max) in seconds, used to convert h_fn's scaled output to days (MODIFIED, see
    module docstring -- OpenXAI's ROS reference quantity is model.predict_with_logits(x) directly, no
    unit conversion needed there).
    p_norm: passed straight to compute_Lp_norm_diff; OpenXAI's own default is 2 (Frobenius/Euclidean).

    Returns {"RIS": {"max": float, "mean": float, "median": float, "samples": [float]},
             "ROS": {"max": float, "mean": float, "median": float, "samples": [float]}}
    -- max/mean/median are np.nan if NO perturbation produced a valid (non-discarded) measure, matching
    OpenXAI's own "return max_measure if valid_perturb else np.nan" convention at the per-instance level."""
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

        # RIS
        repr_diff_ris = compute_Lp_norm_diff(x_repr, xprime_repr, p_norm, normalize_to_relative_change=True)
        if not (np.isnan(repr_diff_ris) or np.isnan(exp_diff) or repr_diff_ris == 0):
            ris_samples.append(exp_diff / repr_diff_ris)

        # ROS
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


# ==== Directly from OpenXAI (openxai/metrics.py, _single_k_gt_faith's 'overlap'/'rank' branches), MODIFIED where noted ==== #

def _topk_idx(attribution, k):
    """Directly from OpenXAI (_construct_topk_dfs): top-k feature indices by descending |attribution|,
    via plain argsort (ties broken by index order, exactly np.argsort's own default behaviour, same as
    OpenXAI's `np.argsort(-np.abs(attr), axis=1)[:, :k]`)."""
    return np.argsort(-np.abs(attribution))[:k]


def _dense_rank(attribution):
    """Directly from OpenXAI (_construct_topk_dfs): `rankdata(-np.abs(attr), method='dense')` -- unlike
    argsort, tied attribution values get the SAME rank (dense method), instead of an arbitrary tie-break
    by index. Only used for Rank Agreement, where OpenXAI compares a feature's rank value, not just its
    top-k position, to tell apart a genuine same-rank match from two different features that merely
    landed in the same top-k POSITION by argsort's tie-breaking happenstance."""
    return rankdata(-np.abs(attribution), method="dense")


def feature_rank_agreement(e_x, e_x_prime, k):
    """Feature Agreement (FA) and Rank Agreement (RA) between the original and ONE perturbed explanation.
    Directly from OpenXAI (openxai/metrics.py's _construct_topk_dfs + _single_k_gt_faith, 'overlap' and
    'rank' branches), reproduced exactly including its tie-handling via dense ranks.

    FA = |topk(e_x) ∩ topk(e_x')| / k (set overlap, argsort-based, ties broken arbitrarily by index --
    exactly OpenXAI's 'overlap' branch).
    RA = fraction of top-k POSITIONS where both the feature index AND its dense rank (which handles ties
    properly, unlike argsort) match between e_x and e_x' -- exactly OpenXAI's 'rank' branch, which builds
    a "feat{idx}rank{rank}" string per position and compares those strings; comparing (idx, rank) pairs
    elementwise here is equivalent (and avoids any string-formatting edge cases) to comparing those exact
    strings.

    MODIFIED vs. OpenXAI: there, `ground_truths` is a synthetic reference explanation used to measure
    faithfulness; here, e_x_prime is a perturbed explanation of the SAME instance, used to measure
    stability instead (see module docstring) -- the formula and its tie-handling are otherwise untouched.

    e_x, e_x_prime: explanation matrices (any shape -- flattened internally, same convention as
    compute_Lp_norm_diff).
    k: number of top features to compare (an absolute int; apply utils.convert_k_to_int equivalent
    yourself if you want OpenXAI's float-as-percentage convention).

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
    """Same Monte Carlo pass as relative_stability_montecarlo_v4, extended to ALSO compute Feature
    Agreement / Rank Agreement for each perturbation (shares the same n_samples perturbed copies and the
    same single batched explain_fn/h_fn call -- no extra model-facing cost over relative_stability_montecarlo_v4).

    MODIFIED vs. OpenXAI (see module docstring and feature_rank_agreement): FA/RA are aggregated here as
    {"min", "mean", "median"} across the n_samples perturbations, not OpenXAI's single ground-truth
    comparison -- min is the "worst case" statistic for an agreement-type metric (higher = more stable),
    the mirror image of RIS/ROS's max convention.

    k_agreement: number of top features to compare for FA/RA (see feature_rank_agreement).
    All other parameters: identical meaning to relative_stability_montecarlo_v4.

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
