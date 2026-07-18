import csv
from datetime import datetime
import gzip
import os
from pathlib import Path
import sys
import tempfile
import time
import zipfile
import xml.etree.ElementTree as ET

import numpy as np

from sklearn.datasets import make_friedman1


JOVO_T7_N_FEATURES = 440_386


def read_label_xlsx(path):
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as zf:
        shared = []
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
        for si in root.findall("a:si", ns):
            shared.append("".join(t.text or "" for t in si.findall(".//a:t", ns)))

        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in sheet.findall(".//a:sheetData/a:row", ns):
            values = []
            for cell in row.findall("a:c", ns):
                value = cell.find("a:v", ns)
                value = "" if value is None else value.text
                if cell.attrib.get("t") == "s" and value:
                    value = shared[int(value)]
                values.append(value)
            rows.append(values)

    header, records = rows[0], rows[1:]
    if header[:3] != ["sample", "patient", "target"]:
        raise ValueError(f"Unexpected label header: {header}")
    return {row[0]: int(float(row[2])) for row in records}


def normalize_sample_id(sample_id):
    return sample_id.replace("_comb", "").removesuffix(".snp")


def read_jovo_t7(data_dir):
    data_dir = Path(data_dir)
    csv_path = data_dir / "t7_20260519_440k.csv.gz"
    label_path = data_dir / "t7_20260519_440k_labels.xlsx"

    labels = read_label_xlsx(label_path)

    t0 = time.perf_counter()
    with gzip.open(csv_path, "rt", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        n_features = len(header) - 1
        X = np.empty((len(labels), n_features), dtype=np.float32)
        y = np.empty(len(labels), dtype=np.int32)
        sample_ids = []
        for row_idx, row in enumerate(reader):
            if row_idx >= len(labels):
                raise ValueError(f"CSV has more rows than labels: row {row_idx + 1}")
            sample_id = normalize_sample_id(row[0])
            if sample_id not in labels:
                raise ValueError(f"Missing label for sample {sample_id}")
            sample_ids.append(sample_id)
            X[row_idx, :] = np.asarray(row[1:], dtype=np.float32)
            y[row_idx] = labels[sample_id]

    missing = sorted(set(sample_ids) - set(labels))
    extra = sorted(set(labels) - set(sample_ids))
    if missing or extra or len(sample_ids) != len(labels):
        raise ValueError(
            f"Label mismatch. missing={missing[:5]} extra={extra[:5]} "
            f"data_rows={len(sample_ids)} label_rows={len(labels)}"
        )

    load_time = time.perf_counter() - t0
    data_args = {
        "dataset": "Jovo T7",
        "csv_path": str(csv_path),
        "label_path": str(label_path),
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "load_time": load_time,
    }
    return X, y, sample_ids, data_args


def make_ydf_dict(X, y=None):
    data = {f"f{i}": X[:, i] for i in range(X.shape[1])}
    if y is not None:
        data["foo"] = y
    return data


def phase(message):
    timestamp = datetime.now().isoformat(timespec="seconds")
    print(f"[phase {timestamp}] {message}", flush=True)


def subsample_jovo_features(x_train, x_test, n_features, seed):
    if n_features is None:
        return x_train, x_test
    n_available = x_train.shape[1]
    if n_features > n_available:
        raise ValueError(f"n_features={n_features} exceeds available {n_available}")
    if n_features == n_available:
        return x_train, x_test
    rng = np.random.default_rng(seed)
    cols = np.sort(rng.permutation(n_available)[:n_features])
    return np.ascontiguousarray(x_train[:, cols]), np.ascontiguousarray(x_test[:, cols])


def run_with_suppressed_native_output(fn, suppress=True):
    if not suppress:
        return fn()

    stdout_fd = 1
    stderr_fd = 2
    saved_stdout = os.dup(stdout_fd)
    saved_stderr = os.dup(stderr_fd)
    with tempfile.TemporaryFile(mode="w+b") as captured:
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            os.dup2(captured.fileno(), stdout_fd)
            os.dup2(captured.fileno(), stderr_fd)
            return fn()
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout, stdout_fd)
            os.dup2(saved_stderr, stderr_fd)
            os.close(saved_stdout)
            os.close(saved_stderr)


def sporf_density_arg(expected_nnz, n_features):
    expected_nnz = float(expected_nnz)
    if expected_nnz.is_integer():
        return int(expected_nnz)
    return expected_nnz / int(n_features)


def sporf_density_fraction(expected_nnz, n_features):
    return float(expected_nnz) / int(n_features)


def make_synthetic_wide_data(
    n_train,
    n_test,
    n_features,
    informative_fraction,
    signal_strength,
    seed,
):
    rng = np.random.default_rng(seed)
    n_total = n_train + n_test
    X = rng.standard_normal((n_total, n_features), dtype=np.float32)
    y = np.arange(n_total, dtype=np.int32) % 2
    rng.shuffle(y)

    n_informative = max(1, int(round(informative_fraction * n_features)))
    class_sign = (2 * y - 1).astype(np.float32)
    X[:, :n_informative] += signal_strength * class_sign[:, None]

    x_train = np.ascontiguousarray(X[:n_train])
    y_train = np.ascontiguousarray(y[:n_train])
    x_test = np.ascontiguousarray(X[n_train:])
    y_test = np.ascontiguousarray(y[n_train:])
    return x_train, y_train, x_test, y_test, n_informative


def friedman1_signal(X):
    return (
        10.0 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20.0 * (X[:, 2] - 0.5) ** 2
        + 10.0 * X[:, 3]
        + 5.0 * X[:, 4]
    )


def make_synthetic_friedman_wide_data(
    n_train,
    n_test,
    n_features,
    informative_fraction,
    noise,
    seed,
    mode,
):
    n_total = n_train + n_test
    if mode == "canonical":
        X, y = make_friedman1(
            n_samples=n_total,
            n_features=n_features,
            noise=noise,
            random_state=seed,
        )
        n_informative = min(5, n_features)
    elif mode == "blocks":
        if n_features < 5:
            raise ValueError("Friedman1 block mode requires at least 5 features")
        X, _ = make_friedman1(
            n_samples=n_total,
            n_features=n_features,
            noise=0.0,
            random_state=seed,
        )
        n_informative = max(5, int(round(informative_fraction * n_features)))
        n_informative = min(n_features, n_informative)
        n_blocks = max(1, n_informative // 5)
        y = np.zeros(n_total, dtype=np.float64)
        for block_idx in range(n_blocks):
            start = 5 * block_idx
            y += friedman1_signal(X[:, start : start + 5])
        y /= np.sqrt(n_blocks)
        if noise:
            rng = np.random.default_rng(seed)
            y += rng.normal(0.0, noise, size=n_total)
    else:
        raise ValueError(f"Unknown Friedman mode: {mode}")

    x_train = np.ascontiguousarray(X[:n_train].astype(np.float32))
    y_train = np.ascontiguousarray(y[:n_train].astype(np.float32))
    x_test = np.ascontiguousarray(X[n_train:].astype(np.float32))
    y_test = np.ascontiguousarray(y[n_train:].astype(np.float32))
    return x_train, y_train, x_test, y_test, n_informative
