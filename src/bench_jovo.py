import argparse
import csv
import gzip
from pathlib import Path
import time
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import ydf

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from cuml.ensemble import SPORFClassifier as sporfc
from cuml.testing.utils import get_handle

DEFAULT_NUM_PROJECTIONS = 5
JOVO_T7_N_FEATURES = 440_386
DEFAULT_MAX_FEATURES = DEFAULT_NUM_PROJECTIONS / JOVO_T7_N_FEATURES


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


def base_hyperparameters(num_features, ntrees, max_features):
    num_trees = ntrees
    num_projections = DEFAULT_NUM_PROJECTIONS
    projection_density = 0.5
    bootstrap_size_ratio = 0.8
    max_depth = 18
    min_samples_leaf = 2
    n_bins = 128

    ydf_args = {
        "label": "foo",
        "bootstrap_size_ratio": bootstrap_size_ratio,
        "max_depth": max_depth,
        "min_examples": min_samples_leaf,
        "num_trees": num_trees,
        "split_axis": "SPARSE_OBLIQUE",
        "sparse_oblique_max_num_projections": num_projections,
        "sparse_oblique_num_projections_exponent": 0.0,
        "sparse_oblique_normalization": "NONE",
        "sparse_oblique_projection_density_factor": projection_density * num_features,
        "sparse_oblique_weights": "BINARY",
    }
    sporf_args = {
        "max_features": max_features,
        "max_samples": bootstrap_size_ratio,
        "density": projection_density,
        "n_bins": n_bins,
        "split_criterion": 0,
        "min_samples_leaf": min_samples_leaf,
        "n_estimators": num_trees,
        "max_leaves": -1,
        "max_depth": max_depth,
        "verbose": False,
    }
    return ydf_args, sporf_args, n_bins


def train_predict_ydf(name, x_train, y_train, x_test, y_test, args, use_slow_engine):
    train_ds = make_ydf_dict(x_train, y_train)
    test_ds = make_ydf_dict(x_test)

    t0 = time.perf_counter()
    model = ydf.RandomForestLearner(**args).train(train_ds)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = model.predict_class(test_ds, use_slow_engine=use_slow_engine).astype(int)
    predict_time = time.perf_counter() - t0

    return {
        "name": name,
        "hyperparameters": args | {"use_slow_engine": use_slow_engine},
        "train_time": train_time,
        "predict_time": predict_time,
        "accuracy": accuracy_score(y_test, pred),
    }


def train_predict_sporf(x_train, y_train, x_test, y_test, args, nstreams):
    n_streams = nstreams
    handle, streams = get_handle(True, n_streams=n_streams)
    args = args | {"handle": handle, "n_streams": n_streams}

    model = sporfc(**args)
    t0 = time.perf_counter()
    model.fit(x_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = model.predict(x_test, predict_model="CPU")
    predict_time = time.perf_counter() - t0

    return {
        "name": "cuML SPORF",
        "hyperparameters": args,
        "train_time": train_time,
        "predict_time": predict_time,
        "accuracy": accuracy_score(y_test, pred),
    }


def print_result(result):
    print(f"{result['name']}")
    print(f"  Hyperparameters: {result['hyperparameters']}")
    print(f"  Training time: {result['train_time']:.2f} seconds")
    print(f"  Prediction time: {result['predict_time']:.2f} seconds")
    print(f"  Test accuracy: {result['accuracy']:.4f}")
    print()


def do_jovo(
    data_dir,
    train_split,
    ydf_use_slow_engine,
    trial,
    ntrees,
    nstreams,
    max_features,
):
    X, y, sample_ids, data_args = read_jovo_t7(data_dir)
    print(f"Loaded {data_args}")
    print(f"Target counts: {dict(zip(*np.unique(y, return_counts=True)))}")

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, train_size=train_split, random_state=123, stratify=y
    )

    x_train = np.ascontiguousarray(x_train.astype(np.float32, copy=False))
    y_train = np.ascontiguousarray(y_train.astype(np.int32, copy=False))
    x_test = np.ascontiguousarray(x_test.astype(np.float32, copy=False))
    y_test = np.ascontiguousarray(y_test.astype(np.int32, copy=False))

    ydf_args, sporf_args, n_bins = base_hyperparameters(
        x_train.shape[1], ntrees, max_features
    )
    ydf_quantized_args = ydf_args | {
        "discretize_numerical_columns": True,
        "num_discretized_numerical_bins": n_bins,
    }

    results = [
        train_predict_sporf(x_train, y_train, x_test, y_test, sporf_args, nstreams)
    ]
    if trial == "all":
        results.extend(
            [
                train_predict_ydf(
                    "YDF sparse oblique",
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    ydf_args,
                    ydf_use_slow_engine,
                ),
                train_predict_ydf(
                    "YDF sparse oblique quantized",
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    ydf_quantized_args,
                    ydf_use_slow_engine,
                ),
            ]
        )

    for result in results:
        print_result(result)
    return results


def parse_max_features(value):
    if value.lower() == "none":
        return None
    try:
        return float(value)
    except ValueError:
        return value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=Path(__file__).resolve().parent.parent / "data/jovo/T7",
        type=Path,
        help="Directory containing the Jovo T7 CSV and label workbook.",
    )
    parser.add_argument(
        "--train-split",
        default=0.8,
        type=float,
        help="Train split fraction.",
    )
    parser.add_argument(
        "--ydf-fast-engine",
        action="store_true",
        help="Use YDF's fast inference engine. The wide T7 data may exceed its projection limits.",
    )
    parser.add_argument(
        "--trial",
        choices=["all", "sporf"],
        default="all",
        help="Trial to run: full comparison or cuML SPORF only.",
    )
    parser.add_argument(
        "--ntrees",
        default=4,
        type=int,
        help="Number of trees for each model.",
    )
    parser.add_argument(
        "--nstreams",
        default=None,
        type=int,
        help="Number of cuML streams. Defaults to --ntrees.",
    )
    parser.add_argument(
        "--max-features",
        default=DEFAULT_MAX_FEATURES,
        type=parse_max_features,
        help=(
            "cuML SPORF max_features passthrough: float, string, or 'None'. "
            f"Defaults to {DEFAULT_MAX_FEATURES:.12g}, equivalent to "
            f"{DEFAULT_NUM_PROJECTIONS} projections on Jovo T7."
        ),
    )
    args = parser.parse_args()
    if args.ntrees < 1:
        parser.error("--ntrees must be at least 1")
    if args.nstreams is None:
        args.nstreams = args.ntrees
    elif args.nstreams < 1:
        parser.error("--nstreams must be at least 1")
    return args


def main():
    args = parse_args()
    do_jovo(
        data_dir=args.data_dir,
        train_split=args.train_split,
        ydf_use_slow_engine=not args.ydf_fast_engine,
        trial=args.trial,
        ntrees=args.ntrees,
        nstreams=args.nstreams,
        max_features=args.max_features,
    )


if __name__ == "__main__":
    main()
