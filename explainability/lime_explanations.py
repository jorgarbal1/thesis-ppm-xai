import numpy as np
from lime.lime_tabular import LimeTabularExplainer

from preprocessing.processor import Task


def _raw_col_index(categorical_columns, numerical_columns, feature_names):
    """{variable name: its column in the raw (max_len, F) array}, categorical and numerical combined."""
    idx = dict(categorical_columns)
    for col in numerical_columns:
        idx[feature_names[col]] = col
    return idx


def build_lime_layout(feature_names, columns_info, categorical_columns, numerical_columns):
    """Builds once (col_index, event_vars, case_vars) and get_layout, a function of length that returns the
    flat-vector layout for that length: event variables get one entry per real position, case variables get
    a single entry regardless of length."""
    col_index = _raw_col_index(categorical_columns, numerical_columns, feature_names)
    event_vars = [name for name in feature_names if columns_info[name] == "event"]
    case_vars = [name for name in feature_names if columns_info[name] == "case"]

    def get_layout(length):
        layout = []
        for t in range(length):
            for name in event_vars:
                layout.append(("event", name, t))
        for name in case_vars:
            layout.append(("case", name, None))
        flat_names = [f"{name}_t{t}" if kind == "event" else name for kind, name, t in layout]
        categorical_flat_idx = [i for i, (kind, name, t) in enumerate(layout) if name in categorical_columns]
        return layout, flat_names, categorical_flat_idx

    return col_index, event_vars, case_vars, get_layout


def flatten_prefix(prefix, length, max_len, layout, col_index):
    """Real prefix (max_len, F) -> flat vector in layout order, padding excluded."""
    start = max_len - length
    out = np.empty(len(layout), dtype = np.float32)
    for i, (kind, name, t) in enumerate(layout):
        col = col_index[name]
        row = start + t if kind == "event" else start
        out[i] = prefix[row, col]
    return out


def unflatten_to_prefix(flat_row, length, max_len, F, layout, col_index):
    """Inverse of flatten_prefix: flat vector -> (max_len, F) array, zero-padded, case variables broadcast
    to every real row."""
    start = max_len - length
    out = np.zeros((max_len, F), dtype = np.float32)
    for i, (kind, name, t) in enumerate(layout):
        col = col_index[name]
        if kind == "event":
            out[start + t, col] = flat_row[i]
        else:
            out[start:, col] = flat_row[i]
    return out


def build_lime_background(X_train, train_k):
    """Groups X_train by exact real prefix length, once.
    Args:
        X_train: load_dataset's own "X_train"
        train_k: load_dataset's own "train_k"

    Returns get_background_raw(length) -> raw rows of that exact length, in raw (unembedded) space (empty
    array if none)."""
    train_len = train_k + 1
    pools = {int(length): X_train[train_len == length] for length in np.unique(train_len)}

    def get_background_raw(length):
        return pools.get(int(length), X_train[:0])

    return get_background_raw


def build_lime_training_data(background_raw, length, max_len, layout, col_index):
    """Flattens a length's background rows (build_lime_background's output) in layout order -- the
    training_data LimeTabularExplainer learns its per-column marginals from (category frequencies, numeric
    quantile bins)."""
    if background_raw.shape[0] == 0:
        raise ValueError(f"empty LIME training pool at length {length}: no real train prefixes of that length")
    return np.stack([flatten_prefix(row, length, max_len, layout, col_index) for row in background_raw], axis = 0)


def explain_lime(model, model_module, task, prefix, length, background_builder, col_index, event_vars, case_vars,
                  get_layout, feature_names, class_idx = None, y_minmax = None, num_samples = 1000, seed = 0):
    if prefix.ndim == 3:
        rows = [explain_lime(model, model_module, task, prefix[b], length, background_builder, col_index,
                              event_vars, case_vars, get_layout, feature_names, class_idx = class_idx,
                              y_minmax = y_minmax, num_samples = num_samples, seed = seed)
                for b in range(prefix.shape[0])]
        return np.stack(rows, axis = 0)

    max_len = prefix.shape[0]
    F = prefix.shape[1]
    layout, flat_names, categorical_flat_idx = get_layout(length)

    background_raw = background_builder(length)
    training_data = build_lime_training_data(background_raw, length, max_len, layout, col_index)
    instance = flatten_prefix(prefix, length, max_len, layout, col_index)

    if task == Task.REMAINING_TIME:
        target_model = model
        pick_col = 0
    else:
        target_model = model_module.build_raw_logits_model(model)
        if class_idx is None:
            logits_x = target_model.predict(prefix[None, ...], verbose = 0)
            pick_col = int(np.argmax(logits_x[0]))
        else:
            pick_col = class_idx

    def predict_fn(flat_rows):
        batch = np.stack([unflatten_to_prefix(row, length, max_len, F, layout, col_index)
                           for row in flat_rows], axis = 0)
        preds = target_model.predict(batch, verbose = 0)
        return preds[:, pick_col]

    explainer = LimeTabularExplainer(training_data, feature_names = flat_names,
                                      categorical_features = categorical_flat_idx,
                                      discretize_continuous = True, mode = "regression",
                                      random_state = seed)
    explanation = explainer.explain_instance(instance, predict_fn, num_features = len(flat_names),
                                              num_samples = num_samples)
    coef_by_idx = dict(explanation.local_exp[1] if isinstance(explanation.local_exp, dict)
                        else explanation.as_map()[1])

    event_matrix = np.zeros((length, len(event_vars)), dtype = np.float64)
    case_scores = {name: 0.0 for name in case_vars}
    for i, (kind, name, t) in enumerate(layout):
        coef = coef_by_idx.get(i, 0.0)
        if kind == "event":
            event_matrix[t, event_vars.index(name)] = coef
        else:
            case_scores[name] = coef

    matrix = np.zeros((length, len(feature_names)), dtype = np.float64)
    for col, name in enumerate(feature_names):
        if name in case_scores:
            matrix[:, col] = case_scores[name]
        else:
            matrix[:, col] = event_matrix[:, event_vars.index(name)]

    if task == Task.REMAINING_TIME:
        y_min, y_max = y_minmax
        matrix = matrix * ((y_max - y_min) / 86400.0)

    return matrix
