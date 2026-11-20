import numpy as np
from alibi.explainers import IntegratedGradients

from preprocessing.processor import Task


def build_ig_central_baseline(model, model_module, X_train, train_k, max_len, column_blocks, categorical_columns):
    """Per-position, per-variable "typical" embedding: median for numerical variables, mean for categorical
    ones (embedding space).
    Args:
        model: the trained model to explain
        model_module: the model's own module (models/lstm/model.py or models/transformer/model.py)
        X_train: load_dataset's own "X_train"
        train_k: load_dataset's own "train_k"
        max_len: load_dataset's own "max_len"
        column_blocks: dict: variable name -> its column slice in embedding space (build_column_blocks)
        categorical_columns: categorical columns

    Returns build_baseline_raw(prefix_len) -> (max_len, E) baseline array, already in embedding space, 0 at
    padding positions."""
    train_len = train_k + 1
    embed_model = model_module.build_embed_model(model)
    z_train = embed_model.predict(X_train, verbose = 0)
    width = z_train.shape[-1]

    event_vals = []
    for e in range(max_len):
        rows = np.where(train_len > e)[0]
        array_pos = max_len - train_len[rows] + e
        event_vals.append(z_train[rows, array_pos])

    event_stat = np.zeros((max_len, width), dtype = np.float32)
    for name, col_slice in column_blocks.items():
        aggregate = np.mean if name in categorical_columns else np.median
        for e in range(max_len):
            event_stat[e, col_slice] = aggregate(event_vals[e][:, col_slice], axis = 0)

    def build_baseline_raw(prefix_len):
        baseline = np.zeros((max_len, width), dtype = np.float32)
        start = max_len - prefix_len
        baseline[start:] = event_stat[:prefix_len]
        return baseline

    return build_baseline_raw


def build_ig_zero_baseline(max_len, width):
    """All-zero baseline, already in embedding space.
    Args:
        max_len: load_dataset's own "max_len"
        width: number of columns in embedding space (build_embed_model's output width, E)

    Returns build_baseline_raw(prefix_len) -> (max_len, E) all-zero array"""
    def build_baseline_raw(prefix_len):
        return np.zeros((max_len, width), dtype = np.float32)

    return build_baseline_raw


def explain_ig(model, model_module, task, prefix, length, baseline_builders, column_blocks, feature_names,
                class_idx = None, y_minmax = None, steps = 100, method = "gausslegendre"):
    """Integrated Gradients. For regression explains the model output transformed to days; for classification,
    the logits.
    Args:
        model: the trained model to explain
        model_module: the model's own module (models/lstm/model.py or models/transformer/model.py)
        task: Task.NEXT_ACTIVITY or Task.REMAINING_TIME
        prefix: (max_len, F) for one prefix, or (B, max_len, F) for a batch sharing the same baseline(s)
        length: real (unpadded) length of the prefix(es) being explained
        baseline_builders: dict: baseline name -> function length -> (max_len, E) baseline array, in embedding space
        column_blocks: dict: variable name -> its column slice in embedding space
        feature_names: variable names, in the order the output matrix should have
        class_idx: classification only: fixed class for the whole batch, or None for each row's own predicted class
        y_minmax: regression only: (y_min, y_max) in seconds the target was scaled with
        steps: Alibi's n_steps
        method: Alibi's integration method ("gausslegendre" default, or any "riemann_*")

    Returns a dict, baseline name -> (matrix, completeness_error_pct). Single-prefix mode gives a (length, F)
    matrix and a float; batch mode gives a (B, length, F) matrix and a (B,) array."""
    batched = prefix.ndim == 3
    prefix_batch = prefix if batched else prefix[None, ...]
    nb_samples = prefix_batch.shape[0]
    max_len = prefix_batch.shape[1]
    start = max_len - length

    embed_model = model_module.build_embed_model(model)
    z_x = embed_model.predict(prefix_batch, verbose = 0)

    if task == Task.REMAINING_TIME:
        target_model = model_module.build_explain_model(model)
        target_arg = None
        pred_x = target_model.predict(z_x, verbose = 0)[:, 0]
    else:
        target_model = model_module.build_logits_model(model)
        logits_x = target_model.predict(z_x, verbose = 0)
        if class_idx is None:
            target_arg = np.argmax(logits_x, axis = 1)
        else:
            target_arg = np.full(nb_samples, class_idx, dtype = int)
        pred_x = logits_x[np.arange(nb_samples), target_arg]

    explainer = IntegratedGradients(target_model, n_steps = steps, method = method)

    results = {}
    for baseline_name, baseline_builder in baseline_builders.items():
        z_b = baseline_builder(length)
        z_b_batch = np.broadcast_to(z_b, z_x.shape)

        if task == Task.REMAINING_TIME:
            pred_b = float(target_model.predict(z_b[None, ...], verbose = 0)[0, 0])
        else:
            logits_b = target_model.predict(z_b[None, ...], verbose = 0)[0]
            pred_b = logits_b[target_arg]
        gap = pred_x - pred_b

        explanation = explainer.explain(z_x, baselines = z_b_batch, target = target_arg)
        attrs_raw = explanation.data["attributions"]
        ig = np.asarray(attrs_raw[0] if isinstance(attrs_raw, list) else attrs_raw)

        sum_ig = ig.sum(axis = (1, 2))
        abs_err = np.abs(gap - sum_ig)
        with np.errstate(divide = "ignore", invalid = "ignore"):
            completeness_error_pct = np.where(gap != 0, abs_err / np.abs(gap) * 100, np.nan)

        matrix = np.stack([ig[:, start:, column_blocks[fname]].sum(axis = -1) for fname in feature_names],
                           axis = -1)

        if task == Task.REMAINING_TIME:
            y_min, y_max = y_minmax
            matrix = matrix * ((y_max - y_min) / 86400.0)

        if not batched:
            results[baseline_name] = (matrix[0], float(completeness_error_pct[0]))
        else:
            results[baseline_name] = (matrix, completeness_error_pct)

    return results
