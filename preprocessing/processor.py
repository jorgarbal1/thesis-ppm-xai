import os
import json
import enum
import pandas as pd
import numpy as np
from multiprocessing import Pool


@enum.unique
class Task(enum.Enum):
    NEXT_ACTIVITY = "next_activity"
    REMAINING_TIME = "remaining_time"


class MultiLogsDataProcessor:
    def __init__(self, name, filepath, core_columns, extra_cat_columns = None, extra_num_columns = None, case_cat_columns = None,
                case_num_columns = None, dir_path = "./datasets/processed", pool = 1):
        """Provides support for processing raw logs.
        Args:
            name: str: Dataset name
            filepath: str: Path to raw logs dataset
            core_columns: list: [case id, activity, timestamp] column names
            extra_cat_columns: list: categorical event attributes [multimodal]
            extra_num_columns: list: numerical event attributes [multimodal]
            case_cat_columns: list: categorical case attributes, repeated in every event [multimodal-sutran]
            case_num_columns: list: numerical case attributes, repeated in every event [multimodal-sutran]
            dir_path:  str: Path to directory for saving the processed dataset
            pool: Number of CPUs (processes) to be used for data processing
        """
        self._name = name
        self._filepath = filepath
        
        self._core_columns = core_columns
        self._extra_cat_columns = list(extra_cat_columns or [])
        self._extra_num_columns = list(extra_num_columns or [])
        self._case_cat_columns = list(case_cat_columns or [])
        self._case_num_columns = list(case_num_columns or [])
        
        self._variable_level = {col: "event" for col in self._extra_cat_columns + self._extra_num_columns}
        self._variable_level.update({col: "case" for col in self._case_cat_columns + self._case_num_columns})
        self._all_cat_columns = self._extra_cat_columns + self._case_cat_columns
        self._all_num_columns = self._extra_num_columns + self._case_num_columns
        self._extra_columns = self._all_cat_columns + self._all_num_columns
        self._all_columns = list(dict.fromkeys(core_columns + self._extra_columns))
        self._extra_out_columns = [f"prefix_{col}" for col in self._extra_columns]
        self._dir_path = dir_path

        if not os.path.exists(f"{dir_path}/{self._name}/processed_multi"):
            os.makedirs(f"{dir_path}/{self._name}/processed_multi")
        self._dir_path = f"{self._dir_path}/{self._name}/processed_multi"
        self._pool = pool

    def _load_df(self, sort_temporally = False):
        """Reads the raw CSV, renames the core columns, cleans the categorical ones, optionally sorts by
        timestamp, propagates case-level columns to every event, and coerces the numerical ones."""
        df = pd.read_csv(self._filepath)
        df = df[self._all_columns]
        df = df.rename(columns = {self._core_columns[0]: "case:concept:name", self._core_columns[1]: "concept:name", self._core_columns[2]: "time:timestamp"})
        df["concept:name"] = df["concept:name"].str.lower().str.replace(" ", "-")
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], format="mixed")

        for col in self._all_cat_columns:
            df[col] = df[col].fillna("[missing]").astype(str).str.lower().str.replace(" ", "-")
        if sort_temporally:
            df.sort_values(by = ["time:timestamp"], kind = "mergesort", inplace = True)

        df["time:timestamp"] = df["time:timestamp"].dt.floor("s")

        first_events = df.drop_duplicates("case:concept:name").set_index("case:concept:name")
        for col in self._case_cat_columns + self._case_num_columns:
            df[col] = df["case:concept:name"].map(first_events[col])

        for col in self._all_num_columns:
            df[col] = pd.to_numeric(df[col], errors = "coerce")
        return df

    def _extract_logs_metadata(self, df, train_list):
        """Builds the vocabularies and numerical stats from the train cases only, and writes them to metadata.json."""
        train_df = df[df["case:concept:name"].isin(train_list)]
        activities = list(train_df["concept:name"].unique())
        coded_activity = {"x_act_dict": {"[PAD]": 0, **{a: i + 1 for i, a in enumerate(activities)}}}
        
        train_targets = train_df[train_df.groupby("case:concept:name").cumcount() > 0]["concept:name"].unique()
        coded_activity["y_act_dict"] = dict(zip(train_targets, range(len(train_targets))))

        coded_activity["extra_dicts"] = {}
        
        for col in self._all_cat_columns:
            values = list(train_df[col].unique())
            cat_keys = ["[PAD]"] + values
            coded_activity["extra_dicts"][col] = dict(zip(cat_keys, range(len(cat_keys))))

        def _stats(values):
            vmin, vmax = float(values.min()), float(values.max())
            return [vmin, vmax if vmax > vmin else vmin + 1.0]

        coded_activity["extra_num_columns"] = self._all_num_columns
        coded_activity["extra_num_stats"] = {col: _stats(train_df[col]) for col in self._all_num_columns}
        coded_activity["time_unit"] = "seconds"
        coded_activity["variable_level"] = self._variable_level
        coded_json = json.dumps(coded_activity)
        with open(f"{self._dir_path}/metadata.json", "w") as metadata_file:
            metadata_file.write(coded_json)

    def _time_feature_sequences(self, times):
        """For each event of the case: time from the case start, time from the previous event (0 for the first), and weekday,
        all in seconds except weekday (its name)."""
        dts = times
        names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        seqs = {"time_from_start": [], "time_from_previous_event": [], "weekday": []}
        for i, t in enumerate(dts):
            seqs["time_from_start"].append(str((t - dts[0]).total_seconds()))
            seqs["time_from_previous_event"].append(str((t - dts[i-1]).total_seconds() if i > 0 else 0.0))
            seqs["weekday"].append(names[t.weekday()])
        return seqs

    def _run_cases(self, rows_fn, df):
        """Runs rows_fn over blocks of whole cases (a case is never split between workers) and keeps the original case order."""
        subs = [sub for _, sub in df.groupby("case:concept:name", sort = False)]
        n_blocks = max(1, min(self._pool, len(subs)))
        if n_blocks == 1:
            return rows_fn(subs)
        bounds = np.linspace(0, len(subs), n_blocks + 1).astype(int)
        blocks = [subs[bounds[i]:bounds[i + 1]] for i in range(n_blocks)]
        with Pool(processes = n_blocks) as pool:
            return pd.concat(pool.map(rows_fn, blocks), ignore_index = True)

    TIME_OUT = ["prefix_time_from_start", "prefix_time_from_previous_event", "prefix_weekday"]

    def _next_activity_rows(self, subs):
        """Builds one row per prefix of each case (every position except the last, which has no next
        activity): the growing prefix of activity/temporal/extra columns, with the next activity as target."""
        cols = ["case_id", "k", "prefix_Activity"] + self.TIME_OUT + self._extra_out_columns + ["next_act"]
        rows = []
        for sub in subs:
            case = sub["case:concept:name"].iloc[0]
            act = sub["concept:name"].to_list()
            time = sub["time:timestamp"].to_list()
            tf_seq = self._time_feature_sequences(time)
            extra = {col: [str(v) for v in sub[col].to_list()] for col in self._extra_columns}
            for i in range(len(act) - 1):
                row = {"case_id": case, "k": i, "prefix_Activity": act[0] if i == 0 else " ".join(act[:i+1])}
                for name, values in tf_seq.items():
                    row[f"prefix_{name}"] = " ".join(values[:i+1])
                for col in self._extra_columns:
                    row[f"prefix_{col}"] = extra[col][0] if i == 0 else " ".join(extra[col][:i+1])
                row["next_act"] = act[i+1]
                rows.append(row)
        return pd.DataFrame(rows, columns = cols)

    def _process_next_activity(self, df, train_list, test_list):
        """Runs _next_activity_rows over all cases, then splits the result into train/test by case_id and writes each to its CSV."""
        processed_df = self._run_cases(self._next_activity_rows, df)
        train_df = processed_df[processed_df["case_id"].isin(train_list)]
        test_df = processed_df[processed_df["case_id"].isin(test_list)]
        train_df.to_csv(f"{self._dir_path}/{Task.NEXT_ACTIVITY.value}_train.csv", index = False)
        test_df.to_csv(f"{self._dir_path}/{Task.NEXT_ACTIVITY.value}_test.csv", index = False)

    def _remaining_time_rows(self, subs):
        """Builds one row per prefix of each case (every position, including the last): the growing prefix of
        activity/temporal/extra columns, with the remaining time in seconds as target."""
        cols = ["case_id", "k", "prefix_Activity"] + self.TIME_OUT + self._extra_out_columns + ["remaining_time_seconds"]
        rows = []
        for sub in subs:
            case = sub["case:concept:name"].iloc[0]
            act = sub["concept:name"].to_list()
            time = sub["time:timestamp"].to_list()
            tf_seq = self._time_feature_sequences(time)
            extra = {col: [str(v) for v in sub[col].to_list()] for col in self._extra_columns}
            for i in range(0, len(act)):
                ttc = time[-1] - time[i]
                row = {"case_id": case, "k": i, "prefix_Activity": act[0] if i == 0 else " ".join(act[:i+1])}
                for name, values in tf_seq.items():
                    row[f"prefix_{name}"] = " ".join(values[:i+1])
                for col in self._extra_columns:
                    row[f"prefix_{col}"] = extra[col][0] if i == 0 else " ".join(extra[col][:i+1])
                row["remaining_time_seconds"] = str(ttc.total_seconds())
                rows.append(row)
        return pd.DataFrame(rows, columns = cols)

    def _process_remaining_time(self, df, train_list, test_list):
        """Runs _remaining_time_rows over all cases, drops the zero-remaining-time rows, then splits the result into train/test by case_id and writes each to its CSV."""
        processed_df = self._run_cases(self._remaining_time_rows, df)
        processed_df = processed_df[pd.to_numeric(processed_df["remaining_time_seconds"]) != 0]
        train_remaining_time = processed_df[processed_df["case_id"].isin(train_list)]
        test_remaining_time = processed_df[processed_df["case_id"].isin(test_list)]
        train_remaining_time.to_csv(f"{self._dir_path}/{Task.REMAINING_TIME.value}_train.csv", index = False)
        test_remaining_time.to_csv(f"{self._dir_path}/{Task.REMAINING_TIME.value}_test.csv", index = False)

    def _strict_temporal_split(self, df, train_test_ratio):
        """Sorts the cases by start time and takes the start time of the case at the train_test_ratio position as
        the cut: train = cases that ended before that cut, test = cases that started at or after it. Cases
        running across the cut are left out of both."""
        span = df.groupby("case:concept:name")["time:timestamp"].agg(["min", "max"])
        span[["min", "max"]] = span[["min", "max"]].apply(pd.to_datetime)
        span = span.sort_values("min", kind = "mergesort")
        n = len(span)
        cut = span["min"].iloc[int(n * train_test_ratio)]
        train_list = span.index[(span["max"] < cut).to_numpy()].to_numpy()
        test_list = span.index[(span["min"] >= cut).to_numpy()].to_numpy()
        print(f"strict temporal split at {cut}: train {len(train_list)} cases, test {len(test_list)} cases, "
              f"left out (crossing the cut) {n - len(train_list) - len(test_list)}")
        return train_list, test_list

    def process_logs(self, task, sort_temporally = False, train_test_ratio = 0.80, train_cases = None, strict_temporal = False):
        """Loads the log, splits it into train/test cases, and writes the processed train/test CSVs and metadata.json for the given task.
        Args:
            task: Task: NEXT_ACTIVITY or REMAINING_TIME
            sort_temporally: bool: sort all events by timestamp before processing (see _load_df)
            train_test_ratio: float: fraction of cases assigned to train; ignored if train_cases is given
            train_cases: list: explicit case ids for train (the rest become test); overrides strict_temporal and train_test_ratio
            strict_temporal: bool: split with _strict_temporal_split (cases crossing the cut are left out) instead of just
                taking the first train_test_ratio fraction of cases
        """
        df = self._load_df(sort_temporally)
        if train_cases is not None:
            all_cases = df["case:concept:name"].unique()
            train_list = np.array([c for c in all_cases if c in set(train_cases)])
            test_list = np.array([c for c in all_cases if c not in set(train_cases)])
        elif strict_temporal:
            train_list, test_list = self._strict_temporal_split(df, train_test_ratio)
        else:
            train_test_ratio = int(abs(df["case:concept:name"].nunique()*train_test_ratio))
            train_list = df["case:concept:name"].unique()[:train_test_ratio]
            test_list = df["case:concept:name"].unique()[train_test_ratio:]
        self._extract_logs_metadata(df, train_list)
        if task == Task.NEXT_ACTIVITY:
            self._process_next_activity(df, train_list, test_list)
        elif task == Task.REMAINING_TIME:
            self._process_remaining_time(df, train_list, test_list)
        else:
            raise ValueError("Invalid task.")