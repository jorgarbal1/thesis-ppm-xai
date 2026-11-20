"""Preprocesses, trains and evaluates the 6 datasets x 2 tasks x 2 architectures (24 combinations) that make up
the final experiment matrix (BPIC12 complete/A/WC, BPIC17 complete, BPIC20 RFP, BPIC20 Permit; next_activity and
remaining_time; LSTM and Transformer).

Resumable at 3 levels (checked in this order, all scoped to this script's own output under processed_datasets/
full_sweep/ and experiments/full_sweep/ -- nothing from any earlier run or repo is ever reused):
  1. results.csv already has a row for (dataset, task, architecture) -> combination skipped entirely.
  2. the checkpoint .keras for (dataset, task, architecture) already exists -> loaded, not retrained.
  3. the processed CSVs for (dataset, task) already exist -> not reprocessed, just loaded.

Usage:
    python scripts/train_full_sweep.py
"""
import csv
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.metrics import mean_absolute_error

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from preprocessing.processor import MultiLogsDataProcessor, Task
from preprocessing import loader
from models.lstm import model as lstm_model
from models.transformer import model as transformer_model

DATA_DIR = os.path.join(REPO, "data")
PROCESSED_ROOT = os.path.join(REPO, "processed_datasets", "full_sweep")
EXPERIMENTS_ROOT = os.path.join(REPO, "experiments", "full_sweep")
RESULTS_CSV = os.path.join(EXPERIMENTS_ROOT, "results.csv")
RESULT_FIELDS = ["dataset", "task", "architecture", "metric_name", "metric_value", "best_epoch",
                  "train_seconds", "n_train", "n_test", "checkpoint_path", "timestamp"]
BY_LENGTH_CSV = os.path.join(EXPERIMENTS_ROOT, "results_by_length.csv")
BY_LENGTH_FIELDS = ["dataset", "task", "architecture", "prefix_length", "metric_name", "metric_value",
                     "n_test", "timestamp"]

CORE_COLUMNS = ["case:concept:name", "concept:name", "time:timestamp"]

DATASETS = {
    "bpic12_complete": dict(raw="bpic2012.csv", filtered="bpic2012_complete.csv",
                             filter_col="lifecycle:transition", filter_val="COMPLETE",
                             extra_cat=["org:resource"], extra_num=[], case_cat=[], case_num=["case:AMOUNT_REQ"]),
    "bpic12_a": dict(filtered="bpic2012_A.csv",
                      extra_cat=["org:resource"], extra_num=[], case_cat=[], case_num=["case:AMOUNT_REQ"]),
    "bpic12_wc": dict(filtered="bpic2012_WC.csv",
                       extra_cat=["org:resource"], extra_num=[], case_cat=[], case_num=["case:AMOUNT_REQ"]),
    "bpic17_complete": dict(raw="bpic2017.csv", filtered="bpic2017_complete.csv",
                             filter_col="lifecycle:transition", filter_val="complete",
                             extra_cat=["org:resource"],
                             extra_num=["FirstWithdrawalAmount", "NumberOfTerms", "MonthlyCost", "CreditScore",
                                        "OfferedAmount"],
                             case_cat=["case:LoanGoal", "case:ApplicationType"],
                             case_num=["case:RequestedAmount"]),
    "bpic20_rfp": dict(filtered="bpic20_requestforpayment.csv",
                        extra_cat=["org:resource", "org:role"], extra_num=[],
                        case_cat=["case:OrganizationalEntity", "case:Project"],
                        case_num=["case:RequestedAmount"]),
    "bpic20_permit": dict(filtered="bpic20_permitlog.csv",
                           extra_cat=["org:resource", "org:role"], extra_num=[],
                           case_cat=["case:OrganizationalEntity", "case:BudgetNumber"],
                           case_num=["case:RequestedBudget"]),
}

ARCH_KWARGS = {
    "lstm": dict(n_neurons=100, n_layers=8),
    "transformer": dict(n_blocks=1, num_heads=4, ff_dim=256, dropout_rate=0.1),
}

EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
BATCH_SIZE = 256


def ensure_filtered_csv(cfg):
    filtered_path = os.path.join(DATA_DIR, cfg["filtered"])
    if os.path.exists(filtered_path):
        return filtered_path
    raw_path = os.path.join(DATA_DIR, cfg["raw"])
    df = pd.read_csv(raw_path, low_memory = False)
    filtered = df[df[cfg["filter_col"]] == cfg["filter_val"]].copy()
    filtered.to_csv(filtered_path, index = False)
    return filtered_path


def processed_dir_ready(processed_dir, task):
    meta = os.path.join(processed_dir, "metadata.json")
    train_csv = os.path.join(processed_dir, f"{task.value}_train.csv")
    test_csv = os.path.join(processed_dir, f"{task.value}_test.csv")
    return os.path.exists(meta) and os.path.exists(train_csv) and os.path.exists(test_csv)


def load_results():
    if not os.path.exists(RESULTS_CSV):
        return []
    with open(RESULTS_CSV, newline = "") as f:
        return list(csv.DictReader(f))


def already_done(results_rows, dataset_key, task, architecture):
    return any(r["dataset"] == dataset_key and r["task"] == task.value and r["architecture"] == architecture
               for r in results_rows)


def append_row(path, fields, row):
    file_exists = os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok = True)
    with open(path, "a", newline = "") as f:
        writer = csv.DictWriter(f, fieldnames = fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def evaluate_by_length(model, data, task):
    """Per-sample error/correctness, plus the same thing grouped by real prefix length (test_k + 1).
    Returns (metric_name, overall_value, {length: (value, n_test_at_that_length)})."""
    lengths = data["test_k"] + 1

    if task == Task.NEXT_ACTIVITY:
        probs = model.predict(data["X_test"], verbose = 0)
        y_true = data["y_test"].argmax(axis = 1)
        y_pred = probs.argmax(axis = 1)
        per_sample = (y_true == y_pred).astype(float)
        metric_name = "accuracy"
    else:
        y_min, y_max = data["y_minmax"]
        to_days = lambda y_scaled: (np.asarray(y_scaled, dtype = float) * (y_max - y_min) + y_min) / 86400.0
        y_pred_days = to_days(model.predict(data["X_test"], verbose = 0).reshape(-1))
        y_true_days = to_days(data["y_test"])
        per_sample = np.abs(y_pred_days - y_true_days)
        metric_name = "mae_days"

    overall = float(per_sample.mean())
    by_length = {int(length): (float(per_sample[lengths == length].mean()), int((lengths == length).sum()))
                 for length in np.unique(lengths)}
    return metric_name, overall, by_length


def main():
    for dataset_key, cfg in DATASETS.items():
        filtered_csv = ensure_filtered_csv(cfg)

        for task in [Task.NEXT_ACTIVITY, Task.REMAINING_TIME]:
            name = f"{dataset_key}_{task.value}"
            processed_dir = os.path.join(PROCESSED_ROOT, name, "processed_multi")

            if processed_dir_ready(processed_dir, task):
                print(f"[preprocess] {name}: already done this run, skipping")
            else:
                print(f"[preprocess] {name}: processing from {filtered_csv}")
                processor = MultiLogsDataProcessor(
                    name = name, filepath = filtered_csv, core_columns = CORE_COLUMNS,
                    extra_cat_columns = cfg["extra_cat"], extra_num_columns = cfg["extra_num"],
                    case_cat_columns = cfg["case_cat"], case_num_columns = cfg["case_num"],
                    dir_path = PROCESSED_ROOT, pool = 4,
                )
                processor.process_logs(task = task, sort_temporally = True, train_test_ratio = 0.8,
                                        strict_temporal = True)

            data = (loader.load_next_activity(processed_dir) if task == Task.NEXT_ACTIVITY
                    else loader.load_remaining_time(processed_dir))
            print(f"[data] {name}: X_train={data['X_train'].shape} X_test={data['X_test'].shape}")

            for architecture in ["lstm", "transformer"]:
                results_rows = load_results()
                if already_done(results_rows, dataset_key, task, architecture):
                    print(f"[skip] {name} {architecture}: already in results.csv")
                    continue

                model_module = lstm_model if architecture == "lstm" else transformer_model
                checkpoint_path = os.path.join(EXPERIMENTS_ROOT, name, f"{task.value}_{architecture}",
                                                "model", f"best_model_{task.value}.keras")

                t0 = time.perf_counter()
                best_epoch = ""
                if os.path.exists(checkpoint_path):
                    print(f"[load] {name} {architecture}: checkpoint already exists, skipping training")
                    model = load_model(checkpoint_path)
                else:
                    print(f"[train] {name} {architecture}")
                    model_name = f"{task.value}_{architecture}_model_{dataset_key}"
                    if task == Task.NEXT_ACTIVITY:
                        model = model_module.build_model(
                            data["max_len"], data["vocab_sizes"], data["categorical_columns"],
                            data["numerical_columns"], task = "classification", output_dim = len(data["classes"]),
                            event_level = False, model_name = model_name, **ARCH_KWARGS[architecture])
                        y_train_fit = data["y_train"]
                    else:
                        model = model_module.build_model(
                            data["max_len"], data["vocab_sizes"], data["categorical_columns"],
                            data["numerical_columns"], task = "regression", model_name = model_name,
                            **ARCH_KWARGS[architecture])
                        y_train_fit = data["y_train"].reshape(-1, 1)

                    history = model_module.train_model(
                        model, data["X_train"], y_train_fit, num_epochs = EPOCHS, batch_size = BATCH_SIZE,
                        checkpoint_path = checkpoint_path, verbose = 2,
                        early_stopping_patience = EARLY_STOPPING_PATIENCE, reduce_lr_patience = REDUCE_LR_PATIENCE)
                    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
                    model = load_model(checkpoint_path)
                train_seconds = time.perf_counter() - t0

                metric_name, metric_value, by_length = evaluate_by_length(model, data, task)
                print(f"[result] {name} {architecture}: {metric_name}={metric_value:.4f} "
                      f"(best_epoch={best_epoch}, {train_seconds:.1f}s)")

                # results_by_length written first: results.csv having this row is what "already_done" trusts to
                # skip re-evaluation on a resumed run, so it must only land once the by-length detail is safely
                # on disk too (otherwise a crash between the two writes could leave an incomplete by-length
                # breakdown permanently marked as done).
                timestamp = datetime.now().isoformat(timespec = "seconds")
                for length in sorted(by_length):
                    value, n_test_at_length = by_length[length]
                    append_row(BY_LENGTH_CSV, BY_LENGTH_FIELDS, {
                        "dataset": dataset_key, "task": task.value, "architecture": architecture,
                        "prefix_length": length, "metric_name": metric_name,
                        "metric_value": f"{value:.6f}", "n_test": n_test_at_length, "timestamp": timestamp,
                    })
                append_row(RESULTS_CSV, RESULT_FIELDS, {
                    "dataset": dataset_key, "task": task.value, "architecture": architecture,
                    "metric_name": metric_name, "metric_value": f"{metric_value:.6f}",
                    "best_epoch": best_epoch, "train_seconds": f"{train_seconds:.1f}",
                    "n_train": data["X_train"].shape[0], "n_test": data["X_test"].shape[0],
                    "checkpoint_path": checkpoint_path, "timestamp": timestamp,
                })

    print("\nAll combinations done.")


if __name__ == "__main__":
    main()
