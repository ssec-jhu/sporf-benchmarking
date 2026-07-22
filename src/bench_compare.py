import argparse
import csv
import gc
import hashlib
import io
import json
import os
from copy import deepcopy
from pathlib import Path
import time

import numpy as np
import ydf

from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from cuml.ensemble import SPORFClassifier
from cuml.ensemble import SPORFRegressor
from cuml.testing.utils import get_handle

from bench_common import JOVO_T7_N_FEATURES
from bench_common import make_synthetic_friedman_wide_data
from bench_common import make_synthetic_wide_data
from bench_common import make_ydf_dict
from bench_common import phase
from bench_common import read_jovo_t7
from bench_common import run_with_suppressed_native_output
from bench_common import sporf_density_arg
from bench_common import sporf_density_fraction


MODEL_LABELS = {
    "cuml": "cuML SPORF",
    "ydf": "YDF sparse oblique",
    "ydf_quantized": "YDF sparse oblique quantized",
}
SUPPORTED_MODELS = set(MODEL_LABELS)
SWEEP_KEYS = {"n_features", "expected_nnz", "n_trees"}
SUPPORTED_DATASETS = {"jovo_t7", "synthetic_wide", "synthetic_friedman"}
SUPPORTED_TASKS = {"classification", "regression"}
SUPPORTED_DIAGNOSTIC_AGGREGATES = {"mean", "median", "sum", "min", "max"}
TIMING_DEFINITION = (
    "train_time excludes data loading/generation, feature subsampling, "
    "and YDF dict construction"
)
TRIAL_SPEC_FILENAME = "trial_spec.json"
PLOT_SPEC_FILENAME = "plot_spec.json"
TRIAL_RESULTS_FILENAME = "results.csv"
TRIAL_RESOLVED_SPEC_FILENAME = "resolved_spec.json"
TRIAL_HPARAMS_FILENAME = "hparams.jsonl"
TRIAL_TREE_DIAGNOSTICS_FILENAME = "tree_diagnostics.csv"
PLOT_OUTPUT_FILENAME = "plot.png"
PLOT_COMBINED_CSV_FILENAME = "combined.csv"
PLOT_RESOLVED_SPEC_FILENAME = "plot.resolved.json"
RESULT_FIELDS = [
    "run_id",
    "spec_name",
    "task",
    "dataset_kind",
    "model",
    "model_label",
    "sweep_key",
    "sweep_value",
    "sweep_index",
    "trial_index",
    "seed",
    "n_train",
    "n_test",
    "n_features",
    "n_trees",
    "n_streams",
    "expected_nnz",
    "density",
    "num_projections",
    "max_depth",
    "min_leaf",
    "bootstrap",
    "n_bins",
    "train_time",
    "predict_time",
    "predict_proba_time",
    "accuracy",
    "log_loss",
    "r2",
    "rmse",
]
TREE_DIAGNOSTIC_FIELDS = [
    "tree_index",
    "treeid",
    "depth",
    "n_nodes",
    "n_split_nodes",
    "n_leaf_nodes",
    "leaf_counter",
    "training_observation_count",
    "weighted_training_path_depth",
    "min_training_leaf_depth",
    "max_training_leaf_depth",
    "train_time_ms",
    "num_outputs",
    "n_projection_slots",
    "n_populated_projection_vectors",
    "projection_indptr_size",
    "projection_indices_size",
    "projection_coeffs_size",
    "projection_payload_nnz",
    "projection_payload_bytes",
    "leaf_vector_size",
    "leaf_vector_bytes",
]
TREE_DIAGNOSTIC_CONTEXT_FIELDS = [
    "run_id",
    "spec_name",
    "task",
    "dataset_kind",
    "model",
    "model_label",
    "sweep_key",
    "sweep_value",
    "sweep_index",
    "trial_index",
    "seed",
    "n_features",
    "n_trees",
    "n_streams",
    "expected_nnz",
    "num_projections",
]


def load_json(path):
    with Path(path).open() as f:
        return json.load(f)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(json_safe(payload), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def canonical_json(payload):
    return json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":"))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(root_dir, path):
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(root_dir) / path


def rmse_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def to_numpy(value):
    if hasattr(value, "to_output"):
        try:
            return np.asarray(value.to_output("numpy"))
        except TypeError:
            return np.asarray(value.to_output())
    if hasattr(value, "get"):
        return np.asarray(value.get())
    return np.asarray(value)


def require_keys(obj, keys, where):
    missing = [key for key in keys if key not in obj]
    if missing:
        raise ValueError(f"{where} missing required key(s): {', '.join(missing)}")


def validate_positive_int(value, name):
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def validate_positive_number(value, name):
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be positive")


def validate_trial_spec(spec):
    require_keys(
        spec,
        ["name", "task", "dataset", "models", "hyperparameters", "trials"],
        "trial spec",
    )
    if spec["task"] not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {spec['task']}")
    dataset = spec["dataset"]
    require_keys(dataset, ["kind"], "dataset")
    if dataset["kind"] not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset kind: {dataset['kind']}")
    if spec["task"] == "regression" and dataset["kind"] not in {"synthetic_friedman"}:
        raise ValueError("Regression currently requires dataset.kind=synthetic_friedman")
    if spec["task"] == "classification" and dataset["kind"] == "synthetic_friedman":
        raise ValueError("synthetic_friedman is a regression dataset")

    models = spec["models"]
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a non-empty list")
    unknown_models = sorted(set(models) - SUPPORTED_MODELS)
    if unknown_models:
        raise ValueError(f"Unsupported model(s): {', '.join(unknown_models)}")

    hparams = spec["hyperparameters"]
    sweep_keys = [
        key for key in SWEEP_KEYS if isinstance(hparams.get(key), list)
    ]
    if len(sweep_keys) != 1:
        raise ValueError(
            "Exactly one of n_features, expected_nnz, n_trees must be a list"
        )
    sweep_key = sweep_keys[0]
    sweep_values = hparams[sweep_key]
    if not sweep_values:
        raise ValueError(f"hyperparameters.{sweep_key} sweep cannot be empty")

    for required in ["n_features", "expected_nnz", "n_trees"]:
        if required not in hparams:
            raise ValueError(f"hyperparameters.{required} is required")
    for required in ["num_projections", "max_depth", "min_leaf", "bootstrap", "n_bins"]:
        if required not in hparams:
            raise ValueError(f"hyperparameters.{required} is required")

    default_n_streams = hparams.get("default_n_streams")
    if default_n_streams is not None:
        validate_positive_int(default_n_streams, "hyperparameters.default_n_streams")

    for idx, item in enumerate(sweep_values):
        if isinstance(item, dict):
            allowed = {"value", "n_streams"}
            unknown = sorted(set(item) - allowed)
            if unknown:
                raise ValueError(
                    f"hyperparameters.{sweep_key}[{idx}] unknown key(s): "
                    + ", ".join(unknown)
                )
            if "value" not in item:
                raise ValueError(f"hyperparameters.{sweep_key}[{idx}] missing value")
            value = item["value"]
            n_streams = item.get("n_streams", default_n_streams)
        else:
            value = item
            n_streams = default_n_streams
        if n_streams is None:
            raise ValueError(
                f"Missing n_streams for {sweep_key}[{idx}]; set per point "
                "or hyperparameters.default_n_streams"
            )
        validate_positive_int(n_streams, f"hyperparameters.{sweep_key}[{idx}].n_streams")
        validate_sweep_value(sweep_key, value, idx)

    for key in ["n_features", "n_trees", "num_projections", "max_depth", "min_leaf", "n_bins"]:
        if key == sweep_key:
            continue
        validate_positive_int(hparams[key], f"hyperparameters.{key}")
    if sweep_key != "expected_nnz":
        validate_positive_number(hparams["expected_nnz"], "hyperparameters.expected_nnz")
    if not 0 < hparams["bootstrap"] <= 1:
        raise ValueError("hyperparameters.bootstrap must be in (0, 1]")

    trials = spec["trials"]
    require_keys(trials, ["n", "base_seed"], "trials")
    validate_positive_int(trials["n"], "trials.n")
    if not isinstance(trials["base_seed"], int):
        raise ValueError("trials.base_seed must be an integer")

    return sweep_key


def validate_sweep_value(key, value, idx):
    if key in {"n_features", "n_trees"}:
        validate_positive_int(value, f"hyperparameters.{key}[{idx}].value")
    elif key == "expected_nnz":
        validate_positive_number(value, f"hyperparameters.{key}[{idx}].value")
    else:
        raise ValueError(f"Unsupported sweep key: {key}")


def resolved_output_paths(root_dir):
    root_dir = Path(root_dir)
    return {
        "root_dir": str(root_dir),
        "results_csv": str(root_dir / TRIAL_RESULTS_FILENAME),
        "resolved_spec_json": str(root_dir / TRIAL_RESOLVED_SPEC_FILENAME),
        "hparams_jsonl": str(root_dir / TRIAL_HPARAMS_FILENAME),
        "tree_diagnostics_csv": str(root_dir / TRIAL_TREE_DIAGNOSTICS_FILENAME),
    }


def seed_grid(base_seed, n_trials):
    rng = np.random.default_rng(base_seed)
    return rng.integers(0, np.iinfo(np.int32).max, size=n_trials, dtype=np.int32).tolist()


def resolve_sweep_points(spec):
    hparams = spec["hyperparameters"]
    sweep_key = validate_trial_spec(spec)
    default_n_streams = hparams.get("default_n_streams")
    fixed = {
        key: deepcopy(value)
        for key, value in hparams.items()
        if key not in {sweep_key, "default_n_streams"}
    }

    points = []
    for idx, item in enumerate(hparams[sweep_key]):
        if isinstance(item, dict):
            value = item["value"]
            n_streams = item.get("n_streams", default_n_streams)
        else:
            value = item
            n_streams = default_n_streams
        resolved = deepcopy(fixed)
        resolved[sweep_key] = value
        resolved["n_streams"] = n_streams
        points.append(
            {
                "sweep_index": idx,
                "sweep_key": sweep_key,
                "sweep_value": value,
                "n_streams": n_streams,
                "hyperparameters": resolved,
            }
        )
    return sweep_key, points


def build_trial_seed_records(spec, points):
    seeds = seed_grid(spec["trials"]["base_seed"], spec["trials"]["n"])
    records = []
    for point in points:
        for trial_index, seed in enumerate(seeds, start=1):
            records.append(
                {
                    "sweep_index": point["sweep_index"],
                    "trial_index": trial_index,
                    "seed": int(seed),
                }
            )
    return records


def learner_args_for_model(task, model, hparams, seed):
    n_features = int(hparams["n_features"])
    expected_nnz = hparams["expected_nnz"]
    n_trees = int(hparams["n_trees"])
    num_projections = int(hparams["num_projections"])
    max_depth = int(hparams["max_depth"])
    min_leaf = int(hparams["min_leaf"])
    bootstrap = float(hparams["bootstrap"])
    n_bins = int(hparams["n_bins"])

    if model == "cuml":
        split_criterion = 0 if task == "classification" else 2
        return {
            "max_features": num_projections,
            "max_samples": bootstrap,
            "density": sporf_density_arg(expected_nnz, n_features),
            "n_bins": n_bins,
            "split_criterion": split_criterion,
            "min_samples_leaf": min_leaf,
            "n_estimators": n_trees,
            "max_leaves": -1,
            "max_depth": max_depth,
            "verbose": False,
            "random_state": int(seed),
        }

    args = {
        "label": "foo",
        "bootstrap_size_ratio": bootstrap,
        "max_depth": max_depth,
        "min_examples": min_leaf,
        "num_trees": n_trees,
        "split_axis": "SPARSE_OBLIQUE",
        "sparse_oblique_max_num_projections": num_projections,
        "sparse_oblique_num_projections_exponent": 1.0,
        "sparse_oblique_normalization": "NONE",
        "sparse_oblique_projection_density_factor": expected_nnz,
        "sparse_oblique_weights": "BINARY",
        "sorting_strategy": "IN_NODE",
        "random_seed": int(seed),
    }
    if task == "regression":
        args["task"] = ydf.Task.REGRESSION
    if model == "ydf_quantized":
        args |= {
            "discretize_numerical_columns": True,
            "num_discretized_numerical_bins": n_bins,
            "sorting_strategy": "PRESORT",
        }
    return args


def build_resolved_spec(spec, source_path, root_dir):
    spec = deepcopy(spec)
    sweep_key, points = resolve_sweep_points(spec)
    outputs = resolved_output_paths(root_dir)
    run_id = spec.get("run_id") or spec["name"]
    trial_records = build_trial_seed_records(spec, points)

    planned_runs = []
    for point in points:
        for trial in trial_records:
            if trial["sweep_index"] != point["sweep_index"]:
                continue
            for model in spec["models"]:
                planned_runs.append(
                    {
                        "run_id": run_id,
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "sweep_key": sweep_key,
                        "sweep_value": point["sweep_value"],
                        "sweep_index": point["sweep_index"],
                        "trial_index": trial["trial_index"],
                        "seed": trial["seed"],
                        "hyperparameters": point["hyperparameters"],
                        "learner_args": learner_args_for_model(
                            spec["task"],
                            model,
                            point["hyperparameters"],
                            trial["seed"],
                        ),
                    }
                )

    return {
        "spec_version": 1,
        "run_id": run_id,
        "root_dir": str(Path(root_dir).resolve()),
        "source_spec": str(source_path),
        "source_spec_sha256": file_sha256(source_path),
        "name": spec["name"],
        "task": spec["task"],
        "dataset": spec["dataset"],
        "models": spec["models"],
        "sweep_key": sweep_key,
        "sweep_points": points,
        "trials": spec["trials"],
        "outputs": outputs,
        "timing_definition": TIMING_DEFINITION,
        "planned_runs": planned_runs,
        "compatibility": compatibility_block(spec, sweep_key),
        "original_spec": spec,
    }


def compatibility_block(spec, sweep_key):
    hparams = spec["hyperparameters"]
    fixed_hparams = {
        key: value
        for key, value in hparams.items()
        if key not in {sweep_key, "default_n_streams"} and not isinstance(value, list)
    }
    return {
        "task": spec["task"],
        "dataset": spec["dataset"],
        "models": sorted(spec["models"]),
        "sweep_key": sweep_key,
        "fixed_hyperparameters": fixed_hparams,
        "timing_definition": TIMING_DEFINITION,
    }


def write_resolved_spec_once(path, resolved):
    path = Path(path)
    if path.exists():
        existing = load_json(path)
        comparable_existing = deepcopy(existing)
        comparable_new = deepcopy(resolved)
        for payload in (comparable_existing, comparable_new):
            payload.pop("source_spec_sha256", None)
        if canonical_json(comparable_existing) != canonical_json(comparable_new):
            raise ValueError(f"Resolved spec already exists and differs: {path}")
        return path
    return write_json(path, resolved)


def write_hparams_jsonl(path, planned_runs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for run in planned_runs:
            f.write(
                "# "
                f"model={run['model']} "
                f"sweep={run['sweep_key']}:{run['sweep_value']} "
                f"trial={run['trial_index']} "
                f"seed={run['seed']} "
                f"n_streams={run['hyperparameters']['n_streams']}\n"
            )
            json.dump(json_safe(run), f, sort_keys=True)
            f.write("\n")
    return path


def default_x_label(sweep_key):
    return {
        "n_features": "Feature dimensionality",
        "expected_nnz": "E[NNZ] per projection",
        "n_trees": "Number of trees",
    }.get(sweep_key, sweep_key)


def default_quality_fields(task):
    if task == "classification":
        return "accuracy", "Test accuracy"
    return "r2", "Test R^2"


def starter_plot_spec(resolved):
    quality_metric, quality_label = default_quality_fields(resolved["task"])
    return {
        "name": resolved["name"],
        "title": resolved["name"],
        "runs": ["."],
        "plot": {
            "x": resolved["sweep_key"],
            "x_label": default_x_label(resolved["sweep_key"]),
            "time_metric": "train_time",
            "quality_metric": quality_metric,
            "quality_label": quality_label,
            "time_scale": "log",
            "models": resolved["models"],
            "diagnostics": {
                "metrics": ["weighted_training_path_depth"],
                "aggregate": "mean",
                "scale": "linear",
                "label": "Expected training observation path depth",
            },
            "caption_note": TIMING_DEFINITION + ".",
        },
    }


def write_starter_plot_spec_once(root_dir, resolved):
    path = Path(root_dir) / PLOT_SPEC_FILENAME
    if path.exists():
        return path
    return write_json(path, starter_plot_spec(resolved))


def write_result_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def parse_tree_diagnostics_csv(csv_text, context):
    if csv_text is None:
        return []
    if isinstance(csv_text, bytes):
        csv_text = csv_text.decode()
    csv_text = str(csv_text).strip()
    if not csv_text:
        return []

    reader = csv.DictReader(io.StringIO(csv_text))
    fieldnames = reader.fieldnames or []
    required = ["tree_index", "treeid"]
    missing = [field for field in required if field not in fieldnames]
    if missing:
        raise ValueError(
            "cuML get_diagnostics_csv() missing expected field(s): "
            + ", ".join(missing)
        )

    rows = []
    for raw in reader:
        row = {field: context.get(field, "") for field in TREE_DIAGNOSTIC_CONTEXT_FIELDS}
        row.update({field: raw.get(field, "") for field in TREE_DIAGNOSTIC_FIELDS})
        rows.append(row)
    return rows


def write_tree_diagnostics_rows(path, rows, append=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if append and path.exists():
        with path.open(newline="") as f:
            existing_fieldnames = csv.DictReader(f).fieldnames or []
        if existing_fieldnames != TREE_DIAGNOSTIC_CONTEXT_FIELDS + TREE_DIAGNOSTIC_FIELDS:
            rows = load_tree_diagnostic_rows(path) + rows
            append = False
    mode = "a" if append else "w"
    with path.open(mode, newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=TREE_DIAGNOSTIC_CONTEXT_FIELDS + TREE_DIAGNOSTIC_FIELDS,
        )
        if not append:
            writer.writeheader()
        writer.writerows(rows)
    return path


def load_result_rows(path):
    rows = []
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(coerce_result_row(row))
    return rows


def coerce_result_row(row):
    int_fields = [
        "sweep_index",
        "trial_index",
        "seed",
        "n_train",
        "n_test",
        "n_features",
        "n_trees",
        "n_streams",
        "num_projections",
        "max_depth",
        "min_leaf",
        "n_bins",
    ]
    float_fields = [
        "sweep_value",
        "expected_nnz",
        "density",
        "bootstrap",
        "train_time",
        "predict_time",
        "predict_proba_time",
        "accuracy",
        "log_loss",
        "r2",
        "rmse",
    ]
    out = dict(row)
    for field in int_fields:
        if out.get(field) not in ("", None):
            out[field] = int(float(out[field]))
    for field in float_fields:
        if out.get(field) not in ("", None):
            out[field] = float(out[field])
        else:
            out[field] = ""
    return out


def load_dataset_source(spec):
    dataset = spec["dataset"]
    root_dir = spec.get("root_dir", ".")
    kind = dataset["kind"]

    if kind != "jovo_t7":
        return None

    if spec["task"] != "classification":
        raise ValueError("Jovo T7 supports classification only")

    phase("Loading Jovo T7 data")
    data_dir = resolve_path(root_dir, dataset.get("data_dir", "data/jovo/T7"))
    X, y, _sample_ids, data_args = read_jovo_t7(data_dir)
    print(f"Loaded {data_args}")
    split_seed = int(dataset.get("split_seed", 123))
    train_idx, test_idx = train_test_split(
        np.arange(y.shape[0]),
        train_size=float(dataset.get("train_split", 0.8)),
        random_state=split_seed,
        stratify=y,
    )
    return {
        "kind": kind,
        "X": X,
        "y": y,
        "train_idx": np.asarray(train_idx),
        "test_idx": np.asarray(test_idx),
    }


def load_dataset_for_point(spec, hparams, seed, dataset_source=None):
    dataset = spec["dataset"]
    kind = dataset["kind"]
    n_features = int(hparams["n_features"])
    task = spec["task"]

    if kind == "synthetic_wide":
        phase("Generating synthetic wide classification data")
        return make_synthetic_wide_data(
            n_train=int(dataset.get("n_train", 800)),
            n_test=int(dataset.get("n_test", 200)),
            n_features=n_features,
            informative_fraction=float(dataset.get("informative_fraction", 0.5)),
            signal_strength=float(dataset.get("signal_strength", 0.5)),
            seed=seed,
        )[:4]

    if kind == "synthetic_friedman":
        phase("Generating synthetic Friedman regression data")
        return make_synthetic_friedman_wide_data(
            n_train=int(dataset.get("n_train", 800)),
            n_test=int(dataset.get("n_test", 200)),
            n_features=n_features,
            informative_fraction=float(dataset.get("informative_fraction", 0.5)),
            noise=float(dataset.get("noise", 1.0)),
            seed=seed,
            mode=dataset.get("mode", "blocks"),
        )[:4]

    if kind == "jovo_t7":
        if task != "classification":
            raise ValueError("Jovo T7 supports classification only")
        if dataset_source is None:
            dataset_source = load_dataset_source(spec)
        X = dataset_source["X"]
        y = dataset_source["y"]
        train_idx = dataset_source["train_idx"]
        test_idx = dataset_source["test_idx"]
        if n_features != JOVO_T7_N_FEATURES:
            feature_seed = int(dataset.get("feature_subsample_seed", seed))
            rng = np.random.default_rng(feature_seed)
            cols = np.sort(rng.permutation(X.shape[1])[:n_features])
            x_train = X[np.ix_(train_idx, cols)]
            x_test = X[np.ix_(test_idx, cols)]
        else:
            x_train = X[train_idx, :]
            x_test = X[test_idx, :]
        y_train = y[train_idx]
        y_test = y[test_idx]
        x_train = np.ascontiguousarray(x_train.astype(np.float32, copy=False))
        y_train = np.ascontiguousarray(y_train.astype(np.int32, copy=False))
        x_test = np.ascontiguousarray(x_test.astype(np.float32, copy=False))
        y_test = np.ascontiguousarray(y_test.astype(np.int32, copy=False))
        return x_train, y_train, x_test, y_test

    raise ValueError(f"Unsupported dataset kind: {kind}")


def prepare_ydf_datasets(x_train, y_train, x_test):
    phase("Constructing YDF datasets")
    return make_ydf_dict(x_train, y_train), make_ydf_dict(x_test)


def train_cuml(task, x_train, y_train, x_test, y_test, learner_args, n_streams):
    handle, _streams = get_handle(True, n_streams=n_streams)
    args = learner_args | {"handle": handle, "n_streams": n_streams}
    estimator_cls = SPORFClassifier if task == "classification" else SPORFRegressor

    phase("cuML SPORF: training")
    estimator = estimator_cls(**args)
    t0 = time.perf_counter()
    estimator.fit(x_train, y_train)
    train_time = time.perf_counter() - t0

    phase("cuML SPORF: predicting")
    t0 = time.perf_counter()
    pred = estimator.predict(x_test, predict_model="CPU")
    predict_time = time.perf_counter() - t0
    proba = None
    proba_labels = None
    predict_proba_time = ""
    if task == "classification" and hasattr(estimator, "predict_proba"):
        phase("cuML SPORF: predicting probabilities")
        t0 = time.perf_counter()
        proba = estimator.predict_proba(x_test)
        predict_proba_time = time.perf_counter() - t0
        if hasattr(estimator, "classes_"):
            proba_labels = to_numpy(estimator.classes_)
    diagnostics_csv = ""
    if hasattr(estimator, "get_diagnostics_csv"):
        diagnostics_csv = estimator.get_diagnostics_csv()
    metrics = score_predictions(
        task,
        y_test,
        pred,
        train_time,
        predict_time,
        proba=proba,
        predict_proba_time=predict_proba_time,
        proba_labels=proba_labels,
    )
    del estimator, pred, proba
    gc.collect()
    return metrics, diagnostics_csv


def train_ydf(task, model_label, train_ds, test_ds, y_test, learner_args):
    learner = ydf.RandomForestLearner(**learner_args)
    phase(f"{model_label}: training")
    t0 = time.perf_counter()
    model = run_with_suppressed_native_output(lambda: learner.train(train_ds), True)
    train_time = time.perf_counter() - t0

    use_slow_engine = True
    phase(f"{model_label}: predicting")
    t0 = time.perf_counter()
    if task == "classification":
        pred = model.predict_class(test_ds, use_slow_engine=use_slow_engine).astype(int)
    else:
        pred = np.asarray(model.predict(test_ds, use_slow_engine=use_slow_engine))
    predict_time = time.perf_counter() - t0
    return score_predictions(task, y_test, pred, train_time, predict_time)


def score_predictions(
    task,
    y_test,
    pred,
    train_time,
    predict_time,
    proba=None,
    predict_proba_time="",
    proba_labels=None,
):
    result = {
        "train_time": train_time,
        "predict_time": predict_time,
        "predict_proba_time": predict_proba_time,
        "accuracy": "",
        "log_loss": "",
        "r2": "",
        "rmse": "",
    }
    if task == "classification":
        result["accuracy"] = float(accuracy_score(y_test, pred))
        if proba is not None:
            proba = to_numpy(proba).astype(np.float64, copy=False)
            labels = (
                np.asarray(proba_labels)
                if proba_labels is not None
                else np.arange(proba.shape[1])
            )
            result["log_loss"] = float(log_loss(y_test, proba, labels=labels))
    else:
        pred = np.asarray(pred, dtype=np.float32)
        result["r2"] = float(r2_score(y_test, pred))
        result["rmse"] = rmse_score(y_test, pred)
    return result


def row_from_result(resolved, point, planned, hparams, x_train, x_test, metrics):
    return {
        "run_id": resolved["run_id"],
        "spec_name": resolved["name"],
        "task": resolved["task"],
        "dataset_kind": resolved["dataset"]["kind"],
        "model": planned["model"],
        "model_label": planned["model_label"],
        "sweep_key": point["sweep_key"],
        "sweep_value": point["sweep_value"],
        "sweep_index": point["sweep_index"],
        "trial_index": planned["trial_index"],
        "seed": planned["seed"],
        "n_train": x_train.shape[0],
        "n_test": x_test.shape[0],
        "n_features": hparams["n_features"],
        "n_trees": hparams["n_trees"],
        "n_streams": hparams["n_streams"],
        "expected_nnz": hparams["expected_nnz"],
        "density": sporf_density_fraction(hparams["expected_nnz"], hparams["n_features"]),
        "num_projections": hparams["num_projections"],
        "max_depth": hparams["max_depth"],
        "min_leaf": hparams["min_leaf"],
        "bootstrap": hparams["bootstrap"],
        "n_bins": hparams["n_bins"],
        **metrics,
    }


def tree_diagnostic_context(resolved, point, planned, hparams):
    return {
        "run_id": resolved["run_id"],
        "spec_name": resolved["name"],
        "task": resolved["task"],
        "dataset_kind": resolved["dataset"]["kind"],
        "model": planned["model"],
        "model_label": planned["model_label"],
        "sweep_key": point["sweep_key"],
        "sweep_value": point["sweep_value"],
        "sweep_index": point["sweep_index"],
        "trial_index": planned["trial_index"],
        "seed": planned["seed"],
        "n_features": hparams["n_features"],
        "n_trees": hparams["n_trees"],
        "n_streams": hparams["n_streams"],
        "expected_nnz": hparams["expected_nnz"],
        "num_projections": hparams["num_projections"],
    }


def result_key(row):
    return (
        row["run_id"],
        row["model"],
        int(row["sweep_index"]),
        int(row["trial_index"]),
        int(row["seed"]),
    )


def planned_result_key(planned):
    return (
        planned["run_id"],
        planned["model"],
        int(planned["sweep_index"]),
        int(planned["trial_index"]),
        int(planned["seed"]),
    )


def load_existing_result_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    return load_result_rows(path)


def run_trial_spec(root_dir):
    root_dir = Path(root_dir).resolve()
    spec_path = root_dir / TRIAL_SPEC_FILENAME
    if not spec_path.exists():
        raise FileNotFoundError(f"Trial root must contain {TRIAL_SPEC_FILENAME}: {root_dir}")
    spec = load_json(spec_path)
    resolved = build_resolved_spec(spec, spec_path, root_dir)
    outputs = resolved["outputs"]

    write_resolved_spec_once(outputs["resolved_spec_json"], resolved)
    plot_spec_path = write_starter_plot_spec_once(root_dir, resolved)
    write_hparams_jsonl(outputs["hparams_jsonl"], resolved["planned_runs"])

    rows = load_existing_result_rows(outputs["results_csv"])
    completed = {result_key(row) for row in rows}
    wrote_tree_diagnostics = Path(outputs["tree_diagnostics_csv"]).exists()
    if rows:
        print(f"Resuming from existing results: {len(rows)} completed row(s)")
    point_by_index = {point["sweep_index"]: point for point in resolved["sweep_points"]}
    grouped = {}
    for planned in resolved["planned_runs"]:
        key = (planned["sweep_index"], planned["trial_index"], planned["seed"])
        grouped.setdefault(key, []).append(planned)

    dataset_source = None

    for (sweep_index, trial_index, seed), planned_group in grouped.items():
        pending_group = [
            planned
            for planned in planned_group
            if planned_result_key(planned) not in completed
        ]
        if not pending_group:
            point = point_by_index[sweep_index]
            print(
                "Skipping completed trial: "
                f"{point['sweep_key']}={point['sweep_value']} "
                f"trial={trial_index}/{resolved['trials']['n']} seed={seed}"
            )
            continue
        point = point_by_index[sweep_index]
        hparams = point["hyperparameters"]
        print(
            "Running trial: "
            f"{point['sweep_key']}={point['sweep_value']} "
            f"trial={trial_index}/{resolved['trials']['n']} seed={seed} "
            f"n_streams={hparams['n_streams']}"
        )
        if resolved["dataset"]["kind"] == "jovo_t7" and dataset_source is None:
            dataset_source = load_dataset_source(resolved)
        x_train, y_train, x_test, y_test = load_dataset_for_point(
            resolved, hparams, seed, dataset_source
        )

        needs_ydf = any(run["model"] in {"ydf", "ydf_quantized"} for run in pending_group)
        train_ds = test_ds = None
        if needs_ydf:
            train_ds, test_ds = prepare_ydf_datasets(x_train, y_train, x_test)

        for planned in pending_group:
            model = planned["model"]
            if model == "cuml":
                metrics, diagnostics_csv = train_cuml(
                    resolved["task"],
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    planned["learner_args"],
                    hparams["n_streams"],
                )
                diagnostic_rows = parse_tree_diagnostics_csv(
                    diagnostics_csv,
                    tree_diagnostic_context(resolved, point, planned, hparams),
                )
                if diagnostic_rows:
                    write_tree_diagnostics_rows(
                        outputs["tree_diagnostics_csv"],
                        diagnostic_rows,
                        append=wrote_tree_diagnostics,
                    )
                    wrote_tree_diagnostics = True
            else:
                metrics = train_ydf(
                    resolved["task"],
                    planned["model_label"],
                    train_ds,
                    test_ds,
                    y_test,
                    planned["learner_args"],
                )
            row = row_from_result(
                resolved, point, planned, hparams, x_train, x_test, metrics
            )
            rows.append(row)
            completed.add(planned_result_key(planned))
            print(
                f"  {row['model_label']}: train={row['train_time']:.4f}s "
                f"predict={row['predict_time']:.4f}s "
                f"{metric_summary(row)}"
            )
            write_result_rows(outputs["results_csv"], rows)
        del x_train, y_train, x_test, y_test, train_ds, test_ds
        gc.collect()

    write_result_rows(outputs["results_csv"], rows)
    print(f"Wrote results CSV: {outputs['results_csv']}")
    print(f"Wrote resolved spec: {outputs['resolved_spec_json']}")
    print(f"Wrote learner hyperparameter JSONL: {outputs['hparams_jsonl']}")
    if wrote_tree_diagnostics:
        print(f"Wrote cuML tree diagnostics CSV: {outputs['tree_diagnostics_csv']}")
    print(f"Wrote starter plot spec: {plot_spec_path}")
    phase("Rendering plot")
    run_plot_spec(root_dir)
    return rows


def metric_summary(row):
    if row["task"] == "classification":
        parts = [f"accuracy={row['accuracy']:.4f}"]
        if row.get("predict_proba_time", "") != "":
            parts.append(f"predict_proba={row['predict_proba_time']:.4f}s")
        if row.get("log_loss", "") != "":
            parts.append(f"log_loss={row['log_loss']:.4f}")
        return " ".join(parts)
    return f"r2={row['r2']:.4f} rmse={row['rmse']:.4f}"


def validate_plot_spec(plot_spec):
    require_keys(plot_spec, ["name", "title", "runs", "plot"], "plot spec")
    if not plot_spec["runs"]:
        raise ValueError("plot spec runs cannot be empty")
    for idx, run in enumerate(plot_spec["runs"]):
        if isinstance(run, str):
            continue
        if isinstance(run, dict):
            require_keys(run, ["dir"], f"runs[{idx}]")
            unknown = sorted(set(run) - {"dir", "spec", "results", "diagnostics"})
            if unknown:
                raise ValueError(
                    f"runs[{idx}] unknown key(s): " + ", ".join(unknown)
                )
            continue
        raise ValueError(f"runs[{idx}] must be a directory string or object")
    plot = plot_spec["plot"]
    require_keys(
        plot,
        ["x", "x_label", "time_metric", "quality_metric", "quality_label", "models"],
        "plot",
    )
    diagnostics = plot.get("diagnostics")
    if diagnostics is not None:
        require_keys(diagnostics, ["metrics"], "plot.diagnostics")
        metrics = diagnostics["metrics"]
        if not isinstance(metrics, list) or not metrics:
            raise ValueError("plot.diagnostics.metrics must be a non-empty list")
        unknown = sorted(set(metrics) - set(TREE_DIAGNOSTIC_FIELDS))
        if unknown:
            raise ValueError(
                "Unknown tree diagnostic metric(s): " + ", ".join(unknown)
            )
        aggregate = diagnostics.get("aggregate", "mean")
        if aggregate not in SUPPORTED_DIAGNOSTIC_AGGREGATES:
            raise ValueError(
                "plot.diagnostics.aggregate must be one of: "
                + ", ".join(sorted(SUPPORTED_DIAGNOSTIC_AGGREGATES))
            )
        scale = diagnostics.get("scale", "log")
        if scale not in {"log", "linear"}:
            raise ValueError("plot.diagnostics.scale must be 'log' or 'linear'")


def resolve_plot_run(plot_root, run):
    if isinstance(run, str):
        run_dir = (plot_root / run).resolve()
        return {
            "dir": str(run_dir),
            "spec": str(run_dir / TRIAL_RESOLVED_SPEC_FILENAME),
            "results": str(run_dir / TRIAL_RESULTS_FILENAME),
            "diagnostics": str(run_dir / TRIAL_TREE_DIAGNOSTICS_FILENAME),
        }

    run_dir = (plot_root / run["dir"]).resolve()
    spec_path = run.get("spec", TRIAL_RESOLVED_SPEC_FILENAME)
    results_path = run.get("results", TRIAL_RESULTS_FILENAME)
    spec_path = Path(spec_path)
    results_path = Path(results_path)
    if not spec_path.is_absolute():
        spec_path = run_dir / spec_path
    if not results_path.is_absolute():
        results_path = run_dir / results_path
    diagnostics_path = run.get("diagnostics", TRIAL_TREE_DIAGNOSTICS_FILENAME)
    diagnostics_path = Path(diagnostics_path)
    if not diagnostics_path.is_absolute():
        diagnostics_path = run_dir / diagnostics_path
    return {
        "dir": str(run_dir),
        "spec": str(spec_path),
        "results": str(results_path),
        "diagnostics": str(diagnostics_path),
    }


def load_plot_inputs(plot_spec, plot_root):
    specs = []
    rows = []
    diagnostic_rows = []
    for run in plot_spec["runs"]:
        resolved_run = resolve_plot_run(plot_root, run)
        spec_path = Path(resolved_run["spec"])
        results_path = Path(resolved_run["results"])
        diagnostics_path = Path(resolved_run["diagnostics"])
        spec = load_json(spec_path)
        spec["_plot_input"] = {
            "dir": resolved_run["dir"],
            "spec": str(spec_path),
            "results": str(results_path),
            "diagnostics": str(diagnostics_path),
            "spec_sha256": file_sha256(spec_path),
            "results_sha256": file_sha256(results_path),
            "diagnostics_sha256": (
                file_sha256(diagnostics_path) if diagnostics_path.exists() else ""
            ),
        }
        specs.append(spec)
        for row in load_result_rows(results_path):
            row["_source_spec"] = str(spec_path)
            row["_source_results"] = str(results_path)
            rows.append(row)
        if diagnostics_path.exists():
            for row in load_tree_diagnostic_rows(diagnostics_path):
                row["_source_spec"] = str(spec_path)
                row["_source_results"] = str(results_path)
                row["_source_diagnostics"] = str(diagnostics_path)
                diagnostic_rows.append(row)
    attach_tree_diagnostic_aggregates(rows, diagnostic_rows, plot_spec)
    return specs, rows


def load_tree_diagnostic_rows(path):
    rows = []
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = [
            field
            for field in TREE_DIAGNOSTIC_CONTEXT_FIELDS
            if field not in fieldnames
        ]
        if missing:
            raise ValueError(
                f"{path} missing expected tree diagnostic field(s): "
                + ", ".join(missing)
            )
        for row in reader:
            rows.append(coerce_tree_diagnostic_row(row))
    return rows


def coerce_tree_diagnostic_row(row):
    int_fields = [
        "sweep_index",
        "trial_index",
        "seed",
        "n_features",
        "n_trees",
        "n_streams",
        "num_projections",
    ]
    numeric_fields = ["sweep_value", "expected_nnz"] + TREE_DIAGNOSTIC_FIELDS
    out = dict(row)
    for field in int_fields:
        if out.get(field) not in ("", None):
            out[field] = int(float(out[field]))
    for field in numeric_fields:
        if out.get(field) not in ("", None):
            out[field] = float(out[field])
        else:
            out[field] = ""
    return out


def tree_diagnostic_group_key(row):
    return (
        row["run_id"],
        row["model"],
        row["sweep_index"],
        row["trial_index"],
        row["seed"],
    )


def aggregate_values(values, aggregate):
    values = [float(value) for value in values if value != ""]
    if not values:
        return ""
    values = np.asarray(values, dtype=np.float64)
    if aggregate == "mean":
        return float(np.mean(values))
    if aggregate == "median":
        return float(np.median(values))
    if aggregate == "sum":
        return float(np.sum(values))
    if aggregate == "min":
        return float(np.min(values))
    if aggregate == "max":
        return float(np.max(values))
    raise ValueError(f"Unsupported aggregate: {aggregate}")


def diagnostic_column_name(metric, aggregate):
    return f"tree_{aggregate}_{metric}"


def attach_tree_diagnostic_aggregates(rows, diagnostic_rows, plot_spec):
    diagnostics = plot_spec["plot"].get("diagnostics")
    if not diagnostics:
        return
    metrics = diagnostics["metrics"]
    aggregate = diagnostics.get("aggregate", "mean")
    grouped = {}
    for row in diagnostic_rows:
        grouped.setdefault(tree_diagnostic_group_key(row), []).append(row)

    for row in rows:
        group = grouped.get(tree_diagnostic_group_key(row), [])
        for metric in metrics:
            row[diagnostic_column_name(metric, aggregate)] = aggregate_values(
                [item[metric] for item in group],
                aggregate,
            )


def compatibility_for_plot(specs, plot_spec):
    base = specs[0]["compatibility"]
    base_without_models = {k: v for k, v in base.items() if k != "models"}
    for spec in specs[1:]:
        other = spec["compatibility"]
        other_without_models = {k: v for k, v in other.items() if k != "models"}
        if canonical_json(base_without_models) != canonical_json(other_without_models):
            raise ValueError(
                "Plot inputs are not compatible. Compatibility blocks differ."
            )
    x_key = plot_spec["plot"]["x"]
    if x_key != base["sweep_key"]:
        raise ValueError(f"plot.x={x_key!r} does not match sweep_key={base['sweep_key']!r}")
    combined = deepcopy(base)
    combined["models"] = sorted(
        {model for spec in specs for model in spec["compatibility"]["models"]}
    )
    return combined


def validate_rows_for_plot(rows, compatibility, plot_spec):
    models = set(plot_spec["plot"]["models"])
    unknown = sorted(models - SUPPORTED_MODELS)
    if unknown:
        raise ValueError(f"Unsupported plot model(s): {', '.join(unknown)}")
    row_models = {row["model"] for row in rows}
    missing = sorted(models - row_models)
    if missing:
        print(f"[phase] Plot warning: no rows for model(s): {', '.join(missing)}")

    per_x_streams = {}
    for row in rows:
        if row["model"] not in models:
            continue
        key = row["sweep_value"]
        per_x_streams.setdefault(key, set()).add(row["n_streams"])
    mixed = {key: values for key, values in per_x_streams.items() if len(values) > 1}
    allow_mixed = bool(plot_spec["plot"].get("allow_mixed_streams_per_x", False))
    if mixed and not allow_mixed:
        raise ValueError(
            "Same sweep value has multiple n_streams values: "
            + ", ".join(f"{key}: {sorted(values)}" for key, values in mixed.items())
        )


def write_combined_csv(path, rows):
    write_result_rows(path, [{key: row.get(key, "") for key in RESULT_FIELDS} for row in rows])
    return Path(path)


def configure_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def compact_number(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return f"{value:g}" if isinstance(value, float) else str(value)


def compact_values(values):
    values = sorted(set(values))
    if len(values) == 1:
        return compact_number(values[0])
    if len(values) <= 5:
        return ", ".join(compact_number(value) for value in values)
    return f"{compact_number(values[0])}-{compact_number(values[-1])}"


def make_caption(plot_spec, compatibility, rows):
    fixed = compatibility["fixed_hyperparameters"]
    fixed_parts = [
        f"trees={fixed.get('n_trees')}",
        f"E[NNZ]={fixed.get('expected_nnz')}",
        f"features={fixed.get('n_features')}",
        f"projections/node={fixed.get('num_projections')}",
        f"max_depth={fixed.get('max_depth')}",
        f"min_leaf={fixed.get('min_leaf')}",
        f"bootstrap={fixed.get('bootstrap')}",
        f"n_bins={fixed.get('n_bins')}",
    ]
    fixed_parts = [part for part in fixed_parts if not part.endswith("=None")]
    stream_pairs = sorted(
        {(row["sweep_value"], row["n_streams"]) for row in rows},
        key=lambda item: item[0],
    )
    streams = ", ".join(
        f"{compact_number(value)}->s{streams}" for value, streams in stream_pairs
    )
    trial_counts = {}
    for row in rows:
        trial_counts.setdefault(row["sweep_value"], set()).add(row["trial_index"])
    counts = {key: len(value) for key, value in trial_counts.items()}
    trial_note = f"trials/x={compact_values(list(counts.values()))}"
    if len(set(counts.values())) > 1:
        trial_note += " (varies)"
    note = plot_spec["plot"].get("caption_note", "")
    return (
        f"task={compatibility['task']} | dataset={compatibility['dataset']['kind']} | "
        f"sweep={compatibility['sweep_key']} | fixed: {', '.join(fixed_parts)} | "
        f"n_streams by sweep value: {streams} | {trial_note} | "
        f"{compatibility['timing_definition']}. {note}"
    )


def plot_combined(plot_spec, rows, compatibility):
    configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot = plot_spec["plot"]
    models = [model for model in plot["models"] if any(row["model"] == model for row in rows)]
    model_labels = [MODEL_LABELS[model] for model in models]
    x_values = sorted({row["sweep_value"] for row in rows})
    x_positions = {value: idx for idx, value in enumerate(x_values)}
    colors = {
        "cuml": "tab:blue",
        "ydf": "tab:orange",
        "ydf_quantized": "tab:green",
    }
    width = min(0.22, 0.75 / max(1, len(models)))
    offsets = np.linspace(
        -width * (len(models) - 1) / 2,
        width * (len(models) - 1) / 2,
        len(models),
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for model, offset in zip(models, offsets):
        for axis, metric in [(axes[0], plot["time_metric"]), (axes[1], plot["quality_metric"])]:
            data = []
            positions = []
            for value in x_values:
                values = [
                    row[metric]
                    for row in rows
                    if row["model"] == model and row["sweep_value"] == value
                    and row[metric] != ""
                ]
                if values:
                    data.append(values)
                    positions.append(x_positions[value] + offset)
            if data:
                axis.boxplot(
                    data,
                    positions=positions,
                    widths=width,
                    patch_artist=True,
                    boxprops={
                        "facecolor": colors[model],
                        "alpha": 0.45,
                        "edgecolor": colors[model],
                    },
                    medianprops={"color": "black"},
                    whiskerprops={"color": "black"},
                    capprops={"color": "black"},
                    flierprops={"marker": "", "markersize": 0},
                )

    diagnostic_axis = add_diagnostic_overlay(
        axes[0],
        plot,
        rows,
        models,
        x_values,
        x_positions,
        colors,
    )

    if plot.get("time_scale", "log") == "log":
        axes[0].set_yscale("log")
    axes[0].set_ylabel(plot.get("time_label", "Training time (s)"))
    axes[1].set_ylabel(plot["quality_label"])
    axes[1].set_xlabel(plot["x_label"])
    axes[0].grid(True, which="both", alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[0].set_title(plot_spec["title"])
    axes[1].set_xticks(range(len(x_values)))
    axes[1].set_xticklabels([compact_number(value) for value in x_values], rotation=35, ha="right")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[model], alpha=0.45)
        for model in models
    ]
    if diagnostic_axis is not None:
        diagnostic_handles, diagnostic_labels = diagnostic_axis.get_legend_handles_labels()
        handles += diagnostic_handles
        model_labels += diagnostic_labels
    axes[0].legend(handles, model_labels, loc="best")
    caption = make_caption(plot_spec, compatibility, rows)
    fig.text(0.5, 0.012, caption, ha="center", va="bottom", fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out = Path(plot_spec["outputs"]["png"])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def diagnostic_label(metric):
    return metric.replace("_", " ")


def diagnostic_series_values(rows, model, x_values, column):
    values = []
    for value in x_values:
        sample = [
            row[column]
            for row in rows
            if row["model"] == model
            and row["sweep_value"] == value
            and row.get(column, "") != ""
        ]
        values.append(float(np.median(sample)) if sample else np.nan)
    return values


def add_diagnostic_overlay(axis, plot, rows, models, x_values, x_positions, colors):
    diagnostics = plot.get("diagnostics")
    if not diagnostics:
        return None

    aggregate = diagnostics.get("aggregate", "mean")
    metrics = diagnostics["metrics"]
    diagnostic_axis = axis.twinx()
    line_styles = ["-", "--", ":", "-."]
    marker_styles = ["o", "s", "^", "D", "v", "P"]
    any_series = False

    for metric_index, metric in enumerate(metrics):
        column = diagnostic_column_name(metric, aggregate)
        for model_index, model in enumerate(models):
            values = diagnostic_series_values(rows, model, x_values, column)
            if not any(np.isfinite(values)):
                continue
            any_series = True
            label = (
                f"{MODEL_LABELS[model]} {aggregate} {diagnostic_label(metric)}"
            )
            diagnostic_axis.plot(
                [x_positions[value] for value in x_values],
                values,
                linestyle=line_styles[metric_index % len(line_styles)],
                marker=marker_styles[(metric_index + model_index) % len(marker_styles)],
                color=colors.get(model, "black"),
                linewidth=1.6,
                markersize=4,
                label=label,
            )

    if not any_series:
        diagnostic_axis.remove()
        return None

    if diagnostics.get("scale", "log") == "log":
        diagnostic_axis.set_yscale("log")
    label = diagnostics.get("label")
    if not label:
        metric_text = ", ".join(diagnostic_label(metric) for metric in metrics)
        label = f"Tree diagnostics ({aggregate}: {metric_text})"
    diagnostic_axis.set_ylabel(label)
    diagnostic_axis.grid(False)
    return diagnostic_axis


def resolve_plot_outputs(plot_spec, plot_root):
    outputs = dict(plot_spec.get("outputs", {}))
    outputs.setdefault("png", PLOT_OUTPUT_FILENAME)
    outputs.setdefault("csv", PLOT_COMBINED_CSV_FILENAME)
    outputs.setdefault(
        "resolved_plot_spec",
        PLOT_RESOLVED_SPEC_FILENAME,
    )
    for key in ["png", "csv", "resolved_plot_spec"]:
        outputs[key] = str(resolve_path(plot_root, outputs[key]))
    return outputs


def run_plot_spec(root_dir):
    root_dir = Path(root_dir).resolve()
    plot_spec_path = root_dir / PLOT_SPEC_FILENAME
    if not plot_spec_path.exists():
        raise FileNotFoundError(f"Plot root must contain {PLOT_SPEC_FILENAME}: {root_dir}")
    plot_spec = load_json(plot_spec_path)
    validate_plot_spec(plot_spec)
    plot_spec["outputs"] = resolve_plot_outputs(plot_spec, root_dir)
    specs, rows = load_plot_inputs(plot_spec, root_dir)
    compatibility = compatibility_for_plot(specs, plot_spec)
    validate_rows_for_plot(rows, compatibility, plot_spec)

    selected_models = set(plot_spec["plot"]["models"])
    rows = [row for row in rows if row["model"] in selected_models]
    write_combined_csv(plot_spec["outputs"]["csv"], rows)

    resolved_plot_spec = deepcopy(plot_spec)
    resolved_plot_spec |= {
        "spec_version": 1,
        "source_plot_spec": str(plot_spec_path),
        "source_plot_spec_sha256": file_sha256(plot_spec_path),
        "input_runs": [spec["_plot_input"] for spec in specs],
        "compatibility": compatibility,
        "n_rows": len(rows),
    }
    write_json(plot_spec["outputs"]["resolved_plot_spec"], resolved_plot_spec)
    out = plot_combined(plot_spec, rows, compatibility)
    print(f"Wrote combined CSV: {plot_spec['outputs']['csv']}")
    print(f"Wrote resolved plot spec: {plot_spec['outputs']['resolved_plot_spec']}")
    print(f"Wrote plot: {out}")
    return out


def schema_payload():
    return {
        "commands": {
            "run": "Run exactly ROOT_DIR/trial_spec.json.",
            "plot": "Render exactly ROOT_DIR/plot_spec.json.",
            "schema": "Print this JSON reference.",
        },
        "models": {
            "values": sorted(SUPPORTED_MODELS),
            "labels": MODEL_LABELS,
        },
        "task": {
            "values": sorted(SUPPORTED_TASKS),
        },
        "dataset": {
            "kind": {
                "values": sorted(SUPPORTED_DATASETS),
                "notes": {
                    "jovo_t7": "Classification only. Uses data_dir, train_split, split_seed, and feature_subsample_seed.",
                    "synthetic_wide": "Classification. Uses n_train, n_test, informative_fraction, and signal_strength.",
                    "synthetic_friedman": "Regression. Uses n_train, n_test, mode, informative_fraction, and noise.",
                },
            },
            "synthetic_friedman.mode": {
                "values": ["blocks", "canonical"],
            },
        },
        "hyperparameters": {
            "required": [
                "n_features",
                "n_trees",
                "expected_nnz",
                "num_projections",
                "max_depth",
                "min_leaf",
                "bootstrap",
                "n_bins",
            ],
            "sweepable": sorted(SWEEP_KEYS),
            "sweep_rule": (
                "Exactly one sweepable hyperparameter must be a list. "
                "Sweep list items are raw values or objects with value and "
                "optional n_streams. If n_streams is omitted, "
                "default_n_streams is used."
            ),
            "optional": ["default_n_streams"],
            "sweep_item_object_fields": ["value", "n_streams"],
        },
        "trials": {
            "required": ["n", "base_seed"],
        },
        "trial_outputs": {
            "root_dir_files": [
                TRIAL_SPEC_FILENAME,
                TRIAL_RESULTS_FILENAME,
                TRIAL_RESOLVED_SPEC_FILENAME,
                TRIAL_HPARAMS_FILENAME,
                TRIAL_TREE_DIAGNOSTICS_FILENAME,
                PLOT_SPEC_FILENAME,
            ],
            "note": (
                "run writes plot_spec.json if it does not already exist. "
                "tree_diagnostics.csv is written when cuML get_diagnostics_csv() "
                "is available and cuML models are run."
            ),
        },
        "plot": {
            "required": [
                "x",
                "x_label",
                "time_metric",
                "quality_metric",
                "quality_label",
                "models",
            ],
            "x_values": sorted(SWEEP_KEYS),
            "time_metric_values": ["train_time", "predict_time", "predict_proba_time"],
            "quality_metric_values": ["accuracy", "log_loss", "r2", "rmse"],
            "time_scale_values": ["log", "linear"],
            "optional": [
                "caption_note",
                "time_scale",
                "time_label",
                "allow_mixed_streams_per_x",
                "diagnostics",
            ],
            "diagnostics": {
                "note": (
                    "Optional right-axis overlay on the training-time panel. "
                    "Values are aggregated per forest from tree_diagnostics.csv "
                    "and plotted as median series by sweep value."
                ),
                "fields": {
                    "metrics": TREE_DIAGNOSTIC_FIELDS,
                    "aggregate": sorted(SUPPORTED_DIAGNOSTIC_AGGREGATES),
                    "scale": ["log", "linear"],
                    "label": "Optional y-axis label.",
                },
                "example": {
                    "metrics": ["n_split_nodes"],
                    "aggregate": "mean",
                    "scale": "log",
                    "label": "Mean split nodes per tree",
                },
            },
        },
        "plot_outputs": {
            "root_dir_files": [
                PLOT_SPEC_FILENAME,
                PLOT_OUTPUT_FILENAME,
                PLOT_COMBINED_CSV_FILENAME,
                PLOT_RESOLVED_SPEC_FILENAME,
            ],
            "runs": (
                "Plot spec runs are trial root dirs relative to the plot root. "
                "Object form is allowed as {'dir': '...', 'spec': '...', "
                "'results': '...', 'diagnostics': '...'}."
            ),
        },
    }


def print_schema():
    print(json.dumps(schema_payload(), indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
trial spec example:
  {
    "name": "synthetic_friedman_width",
    "task": "regression",
    "dataset": {
      "kind": "synthetic_friedman",
      "n_train": 2000,
      "n_test": 1000,
      "mode": "blocks",
      "informative_fraction": 0.5,
      "noise": 1.0
    },
    "models": ["cuml", "ydf"],
    "hyperparameters": {
      "n_trees": 100,
      "n_features": [
        {"value": 10000, "n_streams": 10},
        {"value": 400000, "n_streams": 1}
      ],
      "default_n_streams": 10,
      "expected_nnz": 2,
      "num_projections": 10,
      "max_depth": 18,
      "min_leaf": 2,
      "bootstrap": 0.8,
      "n_bins": 128
    },
    "trials": {"n": 10, "base_seed": 20260710}
  }

plot spec example:
  {
    "name": "synthetic_friedman_width_combined",
    "title": "Synthetic Friedman Width Scaling: SPORF vs YDF",
    "runs": ["../friedman_low", "../friedman_high"],
    "plot": {
      "x": "n_features",
      "x_label": "Feature dimensionality",
      "time_metric": "train_time",
      "quality_metric": "r2",
      "quality_label": "Test R^2",
      "time_scale": "log",
      "models": ["cuml", "ydf"],
      "diagnostics": {
        "metrics": ["n_split_nodes"],
        "aggregate": "mean",
        "scale": "log",
        "label": "Mean split nodes per tree"
      },
      "caption_note": "Training time excludes data generation and YDF dict construction."
    }
  }

Use `bench_compare.py schema` for supported JSON enum values and full spec reference.
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("root_dir", type=Path)
    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("root_dir", type=Path)
    subparsers.add_parser("schema")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "run":
        run_trial_spec(args.root_dir)
    elif args.command == "plot":
        run_plot_spec(args.root_dir)
    elif args.command == "schema":
        print_schema()
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
