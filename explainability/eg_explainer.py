import numpy as np
import tensorflow as tf
import shap

from preprocessing.processor import Task


def build_eg_background(X_train, train_k):
    """Groups X_train by exact real prefix length.
    Args:
        X_train: load_dataset's own "X_train"
        train_k: load_dataset's own "train_k"

    Returns get_background_raw(length) -> (n, max_len, F) array of every real train row of that exact
    length, in raw (unembedded) space (empty array if none)."""
    train_len = train_k + 1
    pools = {int(length): X_train[train_len == length] for length in np.unique(train_len)}

    def get_background_raw(length):
        return pools.get(int(length), X_train[:0])

    return get_background_raw


def explain_eg(model, model_module, task, prefix, length, background_builder, column_blocks, feature_names,
               class_idx = None, y_minmax = None, nsamples = 200, batch_size = 50, seed = 0):
    """Expected Gradients. For regression explains the model output transformed to days; for classification,
    the logits.
    Args:
        model: the trained model to explain
        model_module: the model's own module (models/lstm/model.py or models/transformer/model.py)
        task: Task.NEXT_ACTIVITY or Task.REMAINING_TIME
        prefix: (max_len, F) for one prefix, or (B, max_len, F) for a batch sharing the same background
        length: real (unpadded) length of the prefix(es) being explained
        background_builder: build_eg_background's return value, called here with length
        column_blocks: dict: variable name -> its column slice in embedding space
        feature_names: variable names, in the order the output matrix should have
        class_idx: classification only: fixed class for the whole batch, or None for each row's own predicted
            class
        y_minmax: regression only: (y_min, y_max) in seconds the target was scaled with
        nsamples: shap.GradientExplainer's own nsamples (Monte Carlo draws)
        batch_size: shap.GradientExplainer's own batching, performance only
        seed: shap.GradientExplainer's own rseed

    Returns (matrix, completeness_error_pct, mean_sem, pool_size). Single-prefix mode gives a (length, F)
    matrix and floats; batch mode gives a (B, length, F) matrix and (B,) arrays; pool_size stays a single
    int."""
    batched = prefix.ndim == 3
    prefix_batch = prefix if batched else prefix[None, ...]
    nb_samples = prefix_batch.shape[0]
    max_len = prefix_batch.shape[1]
    start = max_len - length

    embed_model = model_module.build_embed_model(model)
    z_x = embed_model.predict(prefix_batch, verbose = 0)

    background_raw = background_builder(length)
    if background_raw.shape[0] == 0:
        raise ValueError(f"empty background pool at length {length}: no real train prefixes of that length")
    z_background = embed_model.predict(background_raw, verbose = 0)

    if task == Task.REMAINING_TIME:
        target_model = model_module.build_explain_model(model)
        pred_x = target_model.predict(z_x, verbose = 0)[:, 0]
        pred_background_mean = float(target_model.predict(z_background, verbose = 0)[:, 0].mean())
        gap = pred_x - pred_background_mean

        explainer = shap.GradientExplainer(target_model, z_background, batch_size = batch_size,
                                            local_smoothing = 0)
        phi, phi_var = explainer.shap_values(z_x, nsamples = nsamples, rseed = seed, return_variances = True)
    else:
        target_model = model_module.build_logits_model(model)
        if class_idx is None:
            ranked_outputs, output_rank_order = 1, "max"
        else:
            ranked_outputs, output_rank_order = np.full((nb_samples, 1), class_idx), "custom"

        explainer = shap.GradientExplainer(target_model, z_background, batch_size = batch_size,
                                            local_smoothing = 0)
        phi, phi_var, ranks = explainer.shap_values(z_x, nsamples = nsamples, rseed = seed,
                                                      return_variances = True, ranked_outputs = ranked_outputs,
                                                      output_rank_order = output_rank_order)
        target_arg = ranks[:, 0]
        logits_x = target_model.predict(z_x, verbose = 0)
        logits_background_mean = target_model.predict(z_background, verbose = 0).mean(axis = 0)
        pred_x = logits_x[np.arange(nb_samples), target_arg]
        pred_background_mean = logits_background_mean[target_arg]
        gap = pred_x - pred_background_mean

    eg = phi[..., 0]
    sem = np.sqrt(phi_var[0])

    matrix = np.stack([eg[:, start:, column_blocks[name]].sum(axis = -1) for name in feature_names], axis = -1)
    sem_matrix = np.stack([sem[:, start:, column_blocks[name]].sum(axis = -1) for name in feature_names], axis = -1)

    sum_eg = matrix.sum(axis = (1, 2))
    abs_err = np.abs(gap - sum_eg)
    with np.errstate(divide = "ignore", invalid = "ignore"):
        completeness_error_pct = np.where(gap != 0, abs_err / np.abs(gap) * 100, np.nan)

    if task == Task.REMAINING_TIME:
        y_min, y_max = y_minmax
        scale = (y_max - y_min) / 86400.0
        matrix = matrix * scale
        sem_matrix = sem_matrix * scale

    mean_sem = sem_matrix.reshape(sem_matrix.shape[0], -1).mean(axis = 1)

    if not batched:
        return matrix[0], float(completeness_error_pct[0]), float(mean_sem[0]), int(background_raw.shape[0])
    return matrix, completeness_error_pct, mean_sem, int(background_raw.shape[0])
