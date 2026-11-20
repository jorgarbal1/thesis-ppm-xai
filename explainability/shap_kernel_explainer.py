import numpy as np
import shap

from preprocessing.processor import Task


def explain_shap(model, model_module, task, prefix, length, feature_names, class_idx = None, y_minmax = None,
                  nsamples = 8000, l1_reg = 0, seed = 0):
    """KernelSHAP against an all-zero baseline. For regression explains the model output transformed to
    days; for classification, the logits.
    Args:
        model: the trained model to explain
        model_module: the model's own module (models/lstm/model.py or models/transformer/model.py)
        task: Task.NEXT_ACTIVITY or Task.REMAINING_TIME
        prefix: (max_len, F) for one prefix, or (B, max_len, F) for a batch
        length: real (unpadded) length of the prefix(es) being explained
        feature_names: column names in the same order as the F axis of prefix
        class_idx: classification only: fixed class for the whole batch, or None for each row's own predicted
            class
        y_minmax: regression only: (y_min, y_max) in seconds the target was scaled with
        nsamples: shap.KernelExplainer's own nsamples
        l1_reg: shap.KernelExplainer's own l1_reg (0 = dense solve, no feature selection)
        seed: random seed for the coalition sampling

    Returns (matrix, gap): single-prefix mode gives a (length, F) matrix and a float; batch mode gives a
    (B, length, F) matrix and a (B,) array. gap is f(x) - f(baseline), for a completeness check against
    matrix.sum()."""
    if prefix.ndim == 3:
        rows, gaps = [], []
        for b in range(prefix.shape[0]):
            matrix_b, gap_b = explain_shap(model, model_module, task, prefix[b], length, feature_names,
                                            class_idx = class_idx, y_minmax = y_minmax, nsamples = nsamples,
                                            l1_reg = l1_reg, seed = seed)
            rows.append(matrix_b)
            gaps.append(gap_b)
        return np.stack(rows, axis = 0), np.array(gaps)

    max_len, F = prefix.shape
    start = max_len - length
    padding_block = prefix[:start].astype(np.float32)

    if task == Task.REMAINING_TIME:
        f_raw = lambda x: model.predict(x, verbose = 0)[:, 0]
    else:
        raw_logits_model = model_module.build_raw_logits_model(model)
        if class_idx is None:
            class_idx = int(np.argmax(raw_logits_model.predict(prefix[None, ...], verbose = 0)[0]))
        f_raw = lambda x: raw_logits_model.predict(x, verbose = 0)[:, class_idx]

    def f(coalitions):
        n = coalitions.shape[0]
        real_part = coalitions.reshape(n, length, F)
        full = np.zeros((n, max_len, F), dtype = np.float32)
        full[:, :start, :] = padding_block[None, ...]
        full[:, start:, :] = real_part
        return f_raw(full)

    x_real = prefix[start:].astype(np.float32).reshape(1, -1)
    baseline_real = np.zeros((1, length * F), dtype = np.float32)

    f_x = float(f_raw(prefix[None, ...])[0])
    f_baseline = float(f_raw(np.zeros((1, max_len, F), dtype = np.float32))[0])
    gap = f_x - f_baseline

    explainer = shap.KernelExplainer(f, baseline_real, seed = seed)
    shap_values = explainer.shap_values(x_real, nsamples = nsamples, l1_reg = l1_reg)
    matrix = np.asarray(shap_values).reshape(length, F)

    if task == Task.REMAINING_TIME:
        y_min, y_max = y_minmax
        matrix = matrix * ((y_max - y_min) / 86400.0)
        gap = gap * ((y_max - y_min) / 86400.0)

    return matrix, gap
