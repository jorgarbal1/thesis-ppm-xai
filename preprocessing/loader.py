import json
from collections import namedtuple
import numpy as np
import pandas as pd

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TIME_COLUMNS = ["time_from_start", "time_from_previous_event"] # seconds, min-max

# Block = one input variable of the model.
# source: CSV column with the sequence
# name: the variable's output column name
# encode: function; takes the sequence as a list of L str tokens (one per event) and returns an (L, 1) array, one row per event
# level: "event" or "case"
# kind: "categorical" (feeds an Embedding in multi_model.py) or "numerical" (used as-is)
Block = namedtuple("Block", ["source", "name", "encode", "level", "kind"])


def _read(processed_dir, task, split):
    return pd.read_csv(f"{processed_dir}/{task}_{split}.csv", dtype=str, keep_default_na=False)


def _read_metadata(processed_dir):
    with open(f"{processed_dir}/metadata.json") as metadata_file:
        return json.load(metadata_file)


def _minmax(vmin, vmax):
    return vmin, (vmax if vmax > vmin else vmin + 1.0)


def _label_block(source, name, levels, level):
    """Label encoding for a categorical variable: 0 is left for padding, 1..len(levels) are the train
    categories, and len(levels)+1 is for any unseen value. Also returns the vocabulary size, for the
    Embedding layer in multi_model.py."""
    index = {lvl: i + 1 for i, lvl in enumerate(levels)}
    unseen = len(levels) + 1
    vocab_size = len(levels) + 2

    def encode(tokens):
        return np.array([[index.get(tok, unseen)] for tok in tokens], dtype = np.float32)

    return Block(source, name, encode, level, "categorical"), vocab_size


def _minmax_block(source, name, vmin, vmax, level):
    """Min-max with the train min/max (values outside that range are not clipped). NaN -> 0."""
    def encode(tokens):
        scaled = (np.array(tokens, dtype = float) - vmin) / (vmax - vmin)
        return np.nan_to_num(scaled, nan = 0.0).reshape(-1, 1).astype(np.float32)

    return Block(source, name, encode, level, "numerical")


def _feature_blocks(meta, train):
    """Builds one Block per input variable, in CSV order (activity, temporal, weekday, extra categorical,
    extra numerical). Also returns each categorical variable's vocabulary size (for its Embedding) and each
    numerical variable's (vmin, vmax)."""
    variable_level = meta["variable_level"]
    blocks, vocab_sizes, numerical_minmax = [], {}, {}

    activity_levels = [k for k in meta["x_act_dict"] if k != "[PAD]"]
    block, vocab_sizes["Activity"] = _label_block("prefix_Activity", "Activity", activity_levels, "event")
    blocks.append(block)

    for col in TIME_COLUMNS:
        values = np.array(" ".join(train[f"prefix_{col}"]).split(), dtype = float)
        vmin, vmax = _minmax(values.min(), values.max())
        numerical_minmax[col] = (vmin, vmax)
        blocks.append(_minmax_block(f"prefix_{col}", col, vmin, vmax, "event"))

    block, vocab_sizes["weekday"] = _label_block("prefix_weekday", "weekday", WEEKDAYS, "event")
    blocks.append(block)

    for col, e_dict in meta["extra_dicts"].items():
        levels = [k for k in e_dict if k != "[PAD]"]
        block, vocab_sizes[col] = _label_block(f"prefix_{col}", col, levels, variable_level.get(col, "event"))
        blocks.append(block)

    for col in meta["extra_num_columns"]:
        vmin, vmax = meta["extra_num_stats"][col]
        numerical_minmax[col] = (vmin, vmax)
        blocks.append(_minmax_block(f"prefix_{col}", col, vmin, vmax, variable_level.get(col, "event")))

    return blocks, vocab_sizes, numerical_minmax


def _encode(df, blocks, max_len):
    """Encodes every row of df (one row per prefix) into the (n_rows, max_len, width) array that the model takes."""
    width = len(blocks)
    X = np.zeros((len(df), max_len, width), dtype = np.float32)
    tokens = {b.source: df[b.source].str.split().tolist() for b in blocks}
    truncated = 0
    for r in range(len(df)):
        matrix = np.concatenate([b.encode(tokens[b.source][r]) for b in blocks], axis = 1)
        if matrix.shape[0] > max_len: # only possible in test: keep the last max_len events
            matrix = matrix[-max_len:]
            truncated += 1
        X[r, max_len - matrix.shape[0]:, :] = matrix
    if truncated:
        print(f"prefixes longer than max_len (train), truncated to their last {max_len} events: {truncated}")
    return X


def _report_unseen(train, test, sources):
    """Test events whose categorical value never appears in train: they get the "unseen" index (see _label_block)."""
    for src in sources:
        seen = set(" ".join(train[src]).split())
        n = sum(tok not in seen for tok in " ".join(test[src]).split())
        if n:
            print(f"{src}: {n} test events with a value not seen in train (encoded as 'unseen')")


def _prepare_features(processed_dir, task):
    """Loads a task's metadata and train/test CSVs, builds the feature blocks, and derives everything shared
    by load_next_activity/load_remaining_time: max_len, feature names/levels, vocab sizes, and the column
    index of every variable in X."""
    meta = _read_metadata(processed_dir)
    train = _read(processed_dir, task, "train")
    test = _read(processed_dir, task, "test")
    blocks, vocab_sizes, numerical_minmax = _feature_blocks(meta, train)
    _report_unseen(train, test, ["prefix_Activity"] + [f"prefix_{c}" for c in meta["extra_dicts"]])
    max_len = int(train["k"].astype(int).max()) + 1
    feature_names = [b.name for b in blocks]
    columns_info = {b.name: b.level for b in blocks}
    categorical_columns, numerical_columns = {}, []
    
    for col, b in enumerate(blocks):
        if b.kind == "categorical":
            categorical_columns[b.name] = col
        else:
            numerical_columns.append(col)
    return (meta, train, test, blocks, max_len, feature_names, columns_info, vocab_sizes, categorical_columns,
            numerical_columns, numerical_minmax)


def load_next_activity(processed_dir):
    """Target next_act: one-hot over y_act_dict (train vocabulary). Test prefixes whose next activity was never seen
    in train are removed (they have no class to be predicted as)."""
    (meta, train, test, blocks, max_len, feature_names, columns_info, vocab_sizes, categorical_columns,
     numerical_columns, numerical_minmax) = _prepare_features(processed_dir, "next_activity")
    y_index = meta["y_act_dict"]

    known = test["next_act"].isin(list(y_index.keys())).to_numpy()
    if not known.all():
        print(f"test prefixes with a next activity not seen in train, removed: {(~known).sum()}")
        test = test[known]

    eye = np.eye(len(y_index), dtype = np.float32)
    classes = sorted(y_index, key = y_index.get)
    return {
        "X_train": _encode(train, blocks, max_len), "y_train": eye[train["next_act"].map(y_index).astype(int).to_numpy()],
        "X_test": _encode(test, blocks, max_len), "y_test": eye[test["next_act"].map(y_index).astype(int).to_numpy()],
        "feature_names": feature_names, "columns_info": columns_info, "max_len": max_len, "classes": classes,
        "train_k": train["k"].astype(int).to_numpy(), "test_k": test["k"].astype(int).to_numpy(),
        "vocab_sizes": vocab_sizes, "categorical_columns": categorical_columns, "numerical_columns": numerical_columns,
        "numerical_minmax": numerical_minmax,
    }


def load_remaining_time(processed_dir):
    """Target remaining_time_seconds: min-max scaled with the train min/max."""
    (meta, train, test, blocks, max_len, feature_names, columns_info, vocab_sizes, categorical_columns,
     numerical_columns, numerical_minmax) = _prepare_features(processed_dir, "remaining_time")

    y_train = train["remaining_time_seconds"].astype(float).to_numpy()
    y_min, y_max = _minmax(y_train.min(), y_train.max())
    scale = lambda y: ((y - y_min) / (y_max - y_min)).astype(np.float32)

    return {
        "X_train": _encode(train, blocks, max_len), "y_train": scale(y_train),
        "X_test": _encode(test, blocks, max_len), "y_test": scale(test["remaining_time_seconds"].astype(float).to_numpy()),
        "feature_names": feature_names, "columns_info": columns_info, "max_len": max_len, "y_minmax": [y_min, y_max],
        "train_k": train["k"].astype(int).to_numpy(), "test_k": test["k"].astype(int).to_numpy(),
        "vocab_sizes": vocab_sizes, "categorical_columns": categorical_columns, "numerical_columns": numerical_columns,
        "numerical_minmax": numerical_minmax,
    }


# To delete
def _load_regression(processed_dir, task, target):
    """Regression target in seconds, min-max scaled with the train min/max. The returned y_minmax undoes the scaling:
    real = scaled * (max - min) + min."""
    (meta, train, test, blocks, max_len, feature_names, columns_info, vocab_sizes, categorical_columns,
     numerical_columns, numerical_minmax) = _prepare_features(processed_dir, task)
    y_train = train[target].astype(float).to_numpy()
    y_min, y_max = _minmax(y_train.min(), y_train.max())
    scale = lambda y: ((y - y_min) / (y_max - y_min)).astype(np.float32)
    return {
        "X_train": _encode(train, blocks, max_len), "y_train": scale(y_train),
        "X_test": _encode(test, blocks, max_len), "y_test": scale(test[target].astype(float).to_numpy()),
        "feature_names": feature_names, "columns_info": columns_info, "max_len": max_len,
        "numerical_minmax": numerical_minmax,
        "y_minmax": [y_min, y_max],
        "train_k": train["k"].astype(int).to_numpy(), "test_k": test["k"].astype(int).to_numpy(),
        "vocab_sizes": vocab_sizes, "categorical_columns": categorical_columns, "numerical_columns": numerical_columns,
    }


# To delete
def load_next_time(processed_dir):
    """Target next_time (seconds), min-max scaled with train min/max."""
    return _load_regression(processed_dir, "next_time", "next_time_seconds")


# To delete
def load_remaining_time2(processed_dir):
    """Target remaining_time_seconds (seconds), min-max scaled with train min/max."""
    return _load_regression(processed_dir, "remaining_time", "remaining_time_seconds")
