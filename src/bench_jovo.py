import argparse
import csv
import json
from pathlib import Path
import textwrap
import time

import numpy as np
import ydf

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from cuml.ensemble import SPORFClassifier as sporfc
from cuml.ensemble import SPORFRegressor as sporfr
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
from bench_common import subsample_jovo_features

DEFAULT_NUM_PROJECTIONS = 5
DEFAULT_GRID_NTREES = 100
DEFAULT_GRID_NSTREAMS = 8
DEFAULT_SYNTHETIC_WIDE_NTREES = 10
DEFAULT_SYNTHETIC_WIDE_NSTREAMS = 8
DEFAULT_SYNTHETIC_FRIEDMAN_WIDE_NTREES = 10
DEFAULT_SYNTHETIC_FRIEDMAN_WIDE_NSTREAMS = 8
DEFAULT_SYNTHETIC_WIDE_FEATURES = [
    100,
    500,
    1_000,
    5_000,
    10_000,
    50_000,
    100_000,
    200_000,
    300_000,
    400_000,
    500_000,
]
DEFAULT_JOVO_WIDE_FEATURES = [
    100,
    500,
    1_000,
    5_000,
    10_000,
    50_000,
    100_000,
    200_000,
    300_000,
    400_000,
    440_386,
]
DEFAULT_JOVO_TREE_SCALE_NTREES = [10, 30, 100, 300]
DEFAULT_JOVO_TREE_SCALE_NSTREAMS = 10
DEFAULT_JOVO_DENSITY_SCALE_EXPECTED_NNZ = [
    128.0,
    256.0,
    512.0,
    1_024.0,
    2_048.0,
    4_096.0,
    8_192.0,
]
DEFAULT_MAX_FEATURES = DEFAULT_NUM_PROJECTIONS


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
        # Use exponent=1 so max_num_projections is the active cap. With
        # exponent=0, YDF tries one projection per node.
        "sparse_oblique_num_projections_exponent": 1.0,
        "sparse_oblique_normalization": "NONE",
        "sparse_oblique_projection_density_factor": projection_density * num_features,
        "sparse_oblique_weights": "BINARY",
        "sorting_strategy": "IN_NODE",
        "num_discretized_numerical_bins": n_bins
    }
    sporf_args = {
        "max_features": num_projections, # max_features,
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
    phase(f"{name}: constructing YDF datasets")
    train_ds = make_ydf_dict(x_train, y_train)
    test_ds = make_ydf_dict(x_test)

    phase(f"{name}: training")
    t0 = time.perf_counter()
    model = ydf.RandomForestLearner(**args).train(train_ds)
    train_time = time.perf_counter() - t0

    phase(f"{name}: predicting")
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

    phase("cuML SPORF: training")
    model = sporfc(**args)
    t0 = time.perf_counter()
    model.fit(x_train, y_train)
    train_time = time.perf_counter() - t0

    phase("cuML SPORF: predicting")
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


def rmse_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def train_predict_sporf_regressor(x_train, y_train, x_test, y_test, args, nstreams):
    n_streams = nstreams
    handle, streams = get_handle(True, n_streams=n_streams)
    args = args | {"handle": handle, "n_streams": n_streams}

    phase("cuML SPORF: training")
    model = sporfr(**args)
    t0 = time.perf_counter()
    model.fit(x_train, y_train)
    train_time = time.perf_counter() - t0

    phase("cuML SPORF: predicting")
    t0 = time.perf_counter()
    pred = np.asarray(model.predict(x_test, predict_model="CPU"), dtype=np.float32)
    predict_time = time.perf_counter() - t0

    return {
        "name": "cuML SPORF",
        "hyperparameters": args,
        "train_time": train_time,
        "predict_time": predict_time,
        "r2": float(r2_score(y_test, pred)),
        "rmse": rmse_score(y_test, pred),
    }


def train_predict_ydf_regressor_prebuilt(
    name,
    train_ds,
    test_ds,
    y_test,
    args,
    use_slow_engine,
    suppress_output=True,
):
    learner = ydf.RandomForestLearner(**args)
    phase(f"{name}: training")
    t0 = time.perf_counter()
    model = run_with_suppressed_native_output(
        lambda: learner.train(train_ds),
        suppress_output,
    )
    train_time = time.perf_counter() - t0

    phase(f"{name}: predicting")
    t0 = time.perf_counter()
    pred = np.asarray(model.predict(test_ds, use_slow_engine=use_slow_engine))
    predict_time = time.perf_counter() - t0

    return {
        "name": name,
        "hyperparameters": args | {"use_slow_engine": use_slow_engine},
        "train_time": train_time,
        "predict_time": predict_time,
        "r2": float(r2_score(y_test, pred)),
        "rmse": rmse_score(y_test, pred),
    }


def train_predict_ydf_prebuilt(
    name,
    train_ds,
    test_ds,
    y_test,
    args,
    use_slow_engine,
    suppress_output=True,
):
    learner = ydf.RandomForestLearner(**args)
    phase(f"{name}: training")
    t0 = time.perf_counter()
    model = run_with_suppressed_native_output(
        lambda: learner.train(train_ds),
        suppress_output,
    )
    train_time = time.perf_counter() - t0

    phase(f"{name}: predicting")
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


def synthetic_feature_grid(min_features, max_features, n_steps):
    values = np.geomspace(min_features, max_features, num=n_steps)
    values = np.unique(np.rint(values).astype(np.int64))
    values[0] = int(min_features)
    values[-1] = int(max_features)
    return [int(value) for value in values]


def write_rows_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "n_features",
        "trial",
        "seed",
        "n_train",
        "n_test",
        "n_trees",
        "n_streams",
        "num_projections",
        "expected_nnz",
        "max_depth",
        "min_leaf",
        "n_bins",
        "density",
        "train_time",
        "predict_time",
        "accuracy",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def write_regression_rows_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "n_features",
        "trial",
        "seed",
        "n_train",
        "n_test",
        "n_trees",
        "n_streams",
        "num_projections",
        "expected_nnz",
        "max_depth",
        "min_leaf",
        "n_bins",
        "density",
        "n_informative",
        "informative_fraction",
        "friedman_mode",
        "noise",
        "train_time",
        "predict_time",
        "r2",
        "rmse",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def write_commented_jsonl(records, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for record in records:
            f.write(
                "# "
                f"model={record.get('model')} "
                f"trial={record.get('trial')} "
                f"seed={record.get('seed')} "
                f"trees={record.get('n_trees')} "
                f"streams={record.get('n_streams')} "
                f"projections/node={record.get('num_projections')} "
                f"E[NNZ]={record.get('expected_nnz')} "
                f"density_fraction={record.get('density_fraction')}\n"
            )
            json.dump(json_safe(record), f, sort_keys=True)
            f.write("\n")
    return output_path


def read_prior_quantized_accuracy(output_csv):
    output_csv = Path(output_csv)
    if not output_csv.exists():
        return None
    values = []
    try:
        with output_csv.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("model") != "YDF sparse oblique quantized":
                    continue
                try:
                    values.append(float(row["accuracy"]))
                except (KeyError, TypeError, ValueError):
                    continue
    except OSError:
        return None
    if not values:
        return None
    return float(np.mean(values))


def read_synthetic_wide_csv(input_csv):
    input_csv = Path(input_csv)
    rows = []
    with input_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("model") == "YDF sparse oblique quantized":
                continue
            row["n_features"] = int(row["n_features"])
            for optional_int in (
                "n_trees",
                "n_streams",
                "num_projections",
                "max_depth",
                "min_leaf",
                "n_bins",
            ):
                if row.get(optional_int) not in ("", None):
                    row[optional_int] = int(float(row[optional_int]))
            if row.get("expected_nnz") not in ("", None):
                row["expected_nnz"] = float(row["expected_nnz"])
            row["train_time"] = float(row["train_time"])
            row["accuracy"] = float(row["accuracy"])
            rows.append(row)
    if not rows:
        raise ValueError(f"No plottable synthetic wide rows found in {input_csv}")
    return rows


def read_synthetic_friedman_wide_csv(input_csv):
    input_csv = Path(input_csv)
    rows = []
    with input_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["n_features"] = int(row["n_features"])
            for optional_int in (
                "n_trees",
                "n_streams",
                "num_projections",
                "max_depth",
                "min_leaf",
                "n_bins",
                "n_informative",
            ):
                if row.get(optional_int) not in ("", None):
                    row[optional_int] = int(float(row[optional_int]))
            for optional_float in (
                "expected_nnz",
                "density",
                "informative_fraction",
                "noise",
            ):
                if row.get(optional_float) not in ("", None):
                    row[optional_float] = float(row[optional_float])
            row["train_time"] = float(row["train_time"])
            row["predict_time"] = float(row["predict_time"])
            row["r2"] = float(row["r2"])
            row["rmse"] = float(row["rmse"])
            rows.append(row)
    if not rows:
        raise ValueError(f"No plottable Friedman wide rows found in {input_csv}")
    return rows


def read_wide_csv(input_csv, include_quantized=True):
    input_csv = Path(input_csv)
    rows = []
    with input_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (
                not include_quantized
                and row.get("model") == "YDF sparse oblique quantized"
            ):
                continue
            row["n_features"] = int(row["n_features"])
            for optional_int in (
                "n_trees",
                "n_streams",
                "num_projections",
                "max_depth",
                "min_leaf",
                "n_bins",
            ):
                if row.get(optional_int) not in ("", None):
                    row[optional_int] = int(float(row[optional_int]))
            if row.get("expected_nnz") not in ("", None):
                row["expected_nnz"] = float(row["expected_nnz"])
            row["train_time"] = float(row["train_time"])
            row["accuracy"] = float(row["accuracy"])
            rows.append(row)
    if not rows:
        raise ValueError(f"No plottable wide rows found in {input_csv}")
    return rows


def configure_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def compact_number(value):
    if isinstance(value, np.integer):
        return str(int(value))
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:g}"
    return str(value)


def compact_values(values, max_values=5):
    values = sorted(set(values))
    if not values:
        return None
    if len(values) == 1:
        return compact_number(values[0])
    if len(values) <= max_values:
        return ", ".join(compact_number(value) for value in values)
    return f"{compact_number(values[0])}-{compact_number(values[-1])}"


def row_value(row, key):
    if key in row and row[key] not in ("", None):
        return row[key]

    hyperparameters = row.get("hyperparameters")
    if not isinstance(hyperparameters, dict):
        return None

    aliases = {
        "n_trees": ("n_estimators", "num_trees"),
        "n_streams": ("n_streams",),
        "num_projections": ("max_features", "sparse_oblique_max_num_projections"),
        "expected_nnz": ("density", "sparse_oblique_projection_density_factor"),
        "max_depth": ("max_depth",),
        "min_leaf": ("min_samples_leaf", "min_examples"),
        "n_bins": ("n_bins", "num_discretized_numerical_bins"),
    }
    for alias in aliases.get(key, (key,)):
        if alias in hyperparameters and hyperparameters[alias] not in ("", None):
            return hyperparameters[alias]
    return None


CAPTION_FIELDS = [
    ("n_trees", "trees"),
    ("n_streams", "streams"),
    ("num_projections", "projections/node"),
    ("expected_nnz", "E[NNZ]"),
    ("max_depth", "max_depth"),
    ("min_leaf", "min_leaf"),
    ("n_bins", "n_bins"),
]


FOOTNOTE_TAGS = [
    "[A]",
    "[B]",
    "[C]",
    "[D]",
    "[E]",
    "[F]",
    "[G]",
    "[H]",
    "[I]",
    "[J]",
    "[K]",
    "[L]",
    "[M]",
    "[N]",
    "[O]",
    "[P]",
]


def normalized_row_value(row, key):
    value = row_value(row, key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def compact_field_value(rows, key):
    values = [
        value
        for row in rows
        for value in [normalized_row_value(row, key)]
        if value is not None
    ]
    return compact_values(values)


def caption_from_rows(rows, note=None, extra=None, x_field=None):
    parts = []
    tick_tags = {}
    variable_fields = []
    for key, label in CAPTION_FIELDS:
        if key == x_field:
            continue
        compact = compact_field_value(rows, key)
        if compact is None:
            continue
        values = {
            normalized_row_value(row, key)
            for row in rows
            if normalized_row_value(row, key) is not None
        }
        if len(values) == 1 or x_field is None:
            parts.append(f"{label}={compact}")
        else:
            variable_fields.append((key, label))

    if variable_fields and x_field is not None:
        x_values = sorted({row[x_field] for row in rows})
        signature_to_tag = {}
        footnotes = []
        for x_value in x_values:
            x_rows = [row for row in rows if row[x_field] == x_value]
            signature_parts = []
            for key, label in variable_fields:
                compact = compact_field_value(x_rows, key)
                if compact is not None:
                    signature_parts.append(f"{label}={compact}")
            signature = tuple(signature_parts)
            if not signature:
                continue
            if signature not in signature_to_tag:
                tag_idx = len(signature_to_tag)
                if tag_idx < len(FOOTNOTE_TAGS):
                    tag = FOOTNOTE_TAGS[tag_idx]
                else:
                    tag = f"[{tag_idx + 1}]"
                signature_to_tag[signature] = tag
                footnotes.append(f"{tag} " + ", ".join(signature_parts))
            tick_tags[x_value] = signature_to_tag[signature]
        parts.extend(footnotes)

    if extra:
        parts.extend(extra)
    if note:
        parts.append(note)
    return " | ".join(parts), tick_tags


def add_caption(fig, caption, bottom=0.012):
    if not caption:
        return
    fig.text(
        0.5,
        bottom,
        textwrap.fill(caption, width=150),
        ha="center",
        va="bottom",
        fontsize=8.5,
        alpha=0.78,
    )


def plot_wide_boxplots(
    rows,
    output_path,
    title,
    models,
    note,
    quality_field="accuracy",
    quality_label="Test accuracy",
    quality_floor=None,
    quality_ceiling=None,
    x_field="n_features",
    x_label="Feature dimensionality",
):
    configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    colors = {
        "cuML SPORF": "#1f77b4",
        "YDF sparse oblique": "#ff7f0e",
        "YDF sparse oblique quantized": "#2ca02c",
    }
    x_values = sorted({row[x_field] for row in rows})
    caption, tick_tags = caption_from_rows(rows, note=note, x_field=x_field)
    if len(models) == 2:
        offsets = [-0.15, 0.15]
        width = 0.24
    elif len(models) == 3:
        offsets = [-0.24, 0.0, 0.24]
        width = 0.2
    else:
        center = (len(models) - 1) / 2
        offsets = [(idx - center) * 0.18 for idx in range(len(models))]
        width = 0.16

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    metrics = [
        ("train_time", "Training time (s)", True),
        (quality_field, quality_label, False),
    ]

    for ax, (metric, ylabel, log_y) in zip(axes, metrics):
        for model, offset in zip(models, offsets):
            data = [
                [
                    row[metric]
                    for row in rows
                    if row["model"] == model and row[x_field] == x_value
                ]
                for x_value in x_values
            ]
            positions = [idx + offset for idx in range(len(x_values))]
            box = ax.boxplot(
                data,
                positions=positions,
                widths=width,
                patch_artist=True,
                showfliers=False,
            )
            for patch in box["boxes"]:
                patch.set_facecolor(colors[model])
                patch.set_alpha(0.55)
            for median in box["medians"]:
                median.set_color("black")
                median.set_linewidth(1.2)

        if log_y:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", axis="y", alpha=0.3)

        if metric == "train_time":
            for idx, x_value in enumerate(x_values):
                cuml_times = [
                    row["train_time"]
                    for row in rows
                    if row["model"] == "cuML SPORF"
                    and row[x_field] == x_value
                ]
                preferred_competitors = [
                    "YDF sparse oblique",
                    *[
                        model
                        for model in models
                        if model not in ("cuML SPORF", "YDF sparse oblique")
                    ],
                ]
                competitor = None
                competitor_median = None
                for candidate in preferred_competitors:
                    competitor_times = [
                        row["train_time"]
                        for row in rows
                        if row["model"] == candidate
                        and row[x_field] == x_value
                    ]
                    if competitor_times:
                        competitor = candidate
                        competitor_median = float(np.median(competitor_times))
                        break
                if not cuml_times or competitor_median is None:
                    continue
                cuml_median = float(np.median(cuml_times))
                if cuml_median <= 0 or competitor_median <= 0:
                    continue
                if cuml_median <= competitor_median:
                    speedup_label = f"cuML\n{competitor_median / cuml_median:.0f}x"
                else:
                    competitor_label = (
                        "YDF-q"
                        if competitor == "YDF sparse oblique quantized"
                        else "YDF"
                    )
                    speedup_label = (
                        f"{competitor_label}\n{cuml_median / competitor_median:.0f}x"
                    )
                y_position = max(min(cuml_times) / 1.8, 1e-6)
                ax.text(
                    idx + offsets[0],
                    y_position,
                    speedup_label,
                    ha="center",
                    va="top",
                    fontsize=8,
                    bbox={
                        "boxstyle": "round,pad=0.18",
                        "facecolor": "white",
                        "edgecolor": colors["cuML SPORF"],
                        "alpha": 0.75,
                    },
                )

    axes[0].set_title(title)
    axes[1].set_xlabel(x_label)
    axes[1].set_xticks(range(len(x_values)))
    axes[1].set_xticklabels(
        [
            f"{value:g} {tick_tags.get(value, '')}".rstrip()
            for value in x_values
        ],
        rotation=35,
        ha="right",
    )
    quality_values = [row[quality_field] for row in rows]
    lower_quality = min(quality_values) - 0.02
    if quality_floor is not None:
        lower_quality = max(quality_floor, lower_quality)
    upper_quality = max(quality_values) + 0.02
    if quality_ceiling is not None:
        upper_quality = min(quality_ceiling, upper_quality)
    axes[1].set_ylim(lower_quality, upper_quality)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[model], alpha=0.55)
        for model in models
    ]
    axes[0].legend(handles, models, loc="best")

    add_caption(fig, caption, bottom=0.012)
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return output_path


def plot_synthetic_wide_boxplots(rows, output_path, quantized_skip_accuracy=None):
    note = "Training time excludes synthetic data generation and YDF dict construction."
    if quantized_skip_accuracy is not None:
        note += (
            " YDF quantized omitted from this plot "
            f"(prior mean accuracy {quantized_skip_accuracy:.3f})."
        )
    else:
        note += " YDF quantized omitted due to poor accuracy on this data."

    return plot_wide_boxplots(
        rows=rows,
        output_path=output_path,
        title="Synthetic Wide Classification: SPORF vs YDF Sparse Oblique",
        models=["cuML SPORF", "YDF sparse oblique"],
        note=note,
        quality_floor=0.9,
        quality_ceiling=1.01,
    )


def plot_synthetic_friedman_wide_boxplots(rows, output_path):
    models = [
        model
        for model in [
            "cuML SPORF",
            "YDF sparse oblique",
        ]
        if any(row["model"] == model for row in rows)
    ]
    note = (
        "Training time excludes Friedman data generation and YDF dict "
        "construction. Friedman block mode uses additive 5-feature Friedman1 "
        "blocks over the requested informative fraction."
    )
    return plot_wide_boxplots(
        rows=rows,
        output_path=output_path,
        title="Synthetic Friedman1 Wide Regression: SPORF vs YDF Sparse Oblique",
        models=models,
        note=note,
        quality_field="r2",
        quality_label="Test R^2",
        quality_ceiling=1.01,
    )


def print_result(result):
    print(f"{result['name']}")
    print(f"  Hyperparameters: {result['hyperparameters']}")
    print(f"  Training time: {result['train_time']:.2f} seconds")
    print(f"  Prediction time: {result['predict_time']:.2f} seconds")
    print(f"  Test accuracy: {result['accuracy']:.4f}")
    print()


def sporf_density_grid(num_features, min_expected_nnz, max_density, n_steps):
    max_expected_nnz = max_density * num_features
    if min_expected_nnz > max_expected_nnz:
        raise ValueError(
            f"min_expected_nnz={min_expected_nnz} exceeds "
            f"max_expected_nnz={max_expected_nnz}"
        )
    expected_nnz = np.geomspace(min_expected_nnz, max_expected_nnz, num=n_steps)
    expected_nnz = np.unique(np.rint(expected_nnz).astype(np.int64))
    expected_nnz[0] = int(min_expected_nnz)
    expected_nnz[-1] = int(round(max_expected_nnz))
    return [
        {
            "expected_nnz": int(value),
            "density": float(value / num_features),
        }
        for value in expected_nnz
    ]


def sporf_projection_grid(min_num_projections, max_num_projections):
    min_power = int(np.ceil(np.log2(min_num_projections)))
    max_power = int(np.floor(np.log2(max_num_projections)))
    return [2**power for power in range(min_power, max_power + 1)]


def sporf_seed_grid(base_seed, n_seeds):
    rng = np.random.default_rng(base_seed)
    return rng.integers(0, np.iinfo(np.int32).max, size=n_seeds, dtype=np.int32).tolist()


def plot_sporf_density_grid(results, output_path):
    configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    expected_nnz = [result["expected_nnz"] for result in results]
    densities = [result["density"] for result in results]
    accuracies = [result["accuracy"] for result in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(expected_nnz, accuracies, marker="o", linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("SPORF density as E[NNZ] per projection")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Jovo T7 cuML SPORF Accuracy vs Sparse Projection Density")
    ax.grid(True, which="both", alpha=0.3)

    density_labels = [f"{density:.2g}" for density in densities]
    for x, y, label in zip(expected_nnz, accuracies, density_labels):
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    ax.text(
        0.99,
        0.01,
        "point labels are density fractions",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        alpha=0.7,
    )

    caption = caption_from_rows(
        results,
        note="Point labels are density fractions. Training time is not shown.",
    )
    add_caption(fig, caption)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_seed_grid(results, output_path, title):
    configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    trial_index = np.arange(1, len(results) + 1)
    seeds = [result["seed"] for result in results]
    accuracies = [result["accuracy"] for result in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(trial_index, accuracies, marker="o", linewidth=2)
    ax.set_xlabel("Seed draw")
    ax.set_ylabel("Test accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(trial_index)
    ax.set_xticklabels([str(seed) for seed in seeds], rotation=45, ha="right")

    mean_accuracy = float(np.mean(accuracies))
    ax.axhline(mean_accuracy, color="tab:orange", linestyle="--", linewidth=1)
    ax.text(
        0.99,
        0.02,
        f"mean={mean_accuracy:.4f}, std={float(np.std(accuracies)):.4f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        alpha=0.8,
    )

    caption = caption_from_rows(
        results,
        note="Seed sweep uses one fixed Jovo T7 train/test split.",
    )
    add_caption(fig, caption)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_sporf_projection_grid(results, output_path):
    configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_projections = [result["num_projections"] for result in results]
    accuracies = [result["accuracy"] for result in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(num_projections, accuracies, marker="o", linewidth=2)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("SPORF projections tried per node")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Jovo T7 cuML SPORF Accuracy vs Projection Count")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xticks(num_projections)
    ax.set_xticklabels([str(value) for value in num_projections])

    caption = caption_from_rows(
        results,
        note="Projection sweep uses one fixed Jovo T7 train/test split.",
    )
    add_caption(fig, caption)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def do_sporf_density_grid(
    data_dir,
    train_split,
    ntrees,
    nstreams,
    density_grid_steps,
    density_grid_min_expected_nnz,
    density_grid_max_density,
    density_grid_num_projections,
    density_grid_output,
):
    phase("Loading Jovo T7 data")
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

    _, base_sporf_args, _ = base_hyperparameters(
        x_train.shape[1], ntrees, density_grid_num_projections
    )
    base_sporf_args = base_sporf_args | {
        "max_features": density_grid_num_projections,
    }

    grid = sporf_density_grid(
        x_train.shape[1],
        density_grid_min_expected_nnz,
        density_grid_max_density,
        density_grid_steps,
    )

    results = []
    for item in grid:
        sporf_args = base_sporf_args | {
            "density": sporf_density_arg(item["expected_nnz"], x_train.shape[1])
        }
        print(
            "Running cuML SPORF density grid point: "
            f"expected_nnz={item['expected_nnz']} "
            f"density={item['density']:.12g} "
            f"num_projections={density_grid_num_projections}"
        )
        result = train_predict_sporf(
            x_train, y_train, x_test, y_test, sporf_args, nstreams
        )
        result |= item
        results.append(result)
        print_result(result)

    output_path = plot_sporf_density_grid(results, density_grid_output)
    print(f"Wrote density grid plot: {output_path}")
    return results


def do_sporf_seed_grid(
    data_dir,
    train_split,
    ntrees,
    nstreams,
    seed_grid_base_seed,
    seed_grid_count,
    seed_grid_expected_nnz,
    seed_grid_num_projections,
    seed_grid_output,
):
    phase("Loading Jovo T7 data")
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

    density = sporf_density_fraction(seed_grid_expected_nnz, x_train.shape[1])
    _, base_sporf_args, _ = base_hyperparameters(
        x_train.shape[1], ntrees, seed_grid_num_projections
    )
    base_sporf_args = base_sporf_args | {
        "density": sporf_density_arg(seed_grid_expected_nnz, x_train.shape[1]),
        "max_features": seed_grid_num_projections,
    }

    seeds = sporf_seed_grid(seed_grid_base_seed, seed_grid_count)

    results = []
    for seed in seeds:
        sporf_args = base_sporf_args | {"random_state": int(seed)}
        print(
            "Running cuML SPORF seed grid point: "
            f"seed={seed} "
            f"expected_nnz={seed_grid_expected_nnz:g} "
            f"density={density:.12g} "
            f"num_projections={seed_grid_num_projections}"
        )
        result = train_predict_sporf(
            x_train, y_train, x_test, y_test, sporf_args, nstreams
        )
        result |= {
            "seed": int(seed),
            "expected_nnz": float(seed_grid_expected_nnz),
            "density": float(density),
            "num_projections": int(seed_grid_num_projections),
        }
        results.append(result)
        print_result(result)

    accuracies = np.asarray([result["accuracy"] for result in results], dtype=np.float64)
    print(
        "Seed grid accuracy summary: "
        f"mean={accuracies.mean():.4f} "
        f"std={accuracies.std():.4f} "
        f"min={accuracies.min():.4f} "
        f"max={accuracies.max():.4f}"
    )

    output_path = plot_seed_grid(
        results,
        seed_grid_output,
        "Jovo T7 cuML SPORF Accuracy vs Pseudorandom Seed",
    )
    print(f"Wrote seed grid plot: {output_path}")
    return results


def do_ydf_seed_grid(
    data_dir,
    train_split,
    ydf_use_slow_engine,
    ntrees,
    seed_grid_base_seed,
    seed_grid_count,
    seed_grid_expected_nnz,
    seed_grid_num_projections,
    ydf_seed_grid_output,
):
    phase("Loading Jovo T7 data")
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

    ydf_args, _, _ = base_hyperparameters(
        x_train.shape[1], ntrees, seed_grid_num_projections
    )
    ydf_args = ydf_args | {
        "sparse_oblique_max_num_projections": seed_grid_num_projections,
        "sparse_oblique_num_projections_exponent": 1.0,
        "sparse_oblique_projection_density_factor": seed_grid_expected_nnz,
    }

    seeds = sporf_seed_grid(seed_grid_base_seed, seed_grid_count)

    results = []
    for seed in seeds:
        args = ydf_args | {"random_seed": int(seed)}
        print(
            "Running YDF sparse-oblique seed grid point: "
            f"seed={seed} "
            f"expected_nnz={seed_grid_expected_nnz:g} "
            f"num_projections={seed_grid_num_projections}"
        )
        result = train_predict_ydf(
            "YDF sparse oblique",
            x_train,
            y_train,
            x_test,
            y_test,
            args,
            ydf_use_slow_engine,
        )
        result |= {
            "seed": int(seed),
            "expected_nnz": float(seed_grid_expected_nnz),
            "num_projections": int(seed_grid_num_projections),
        }
        results.append(result)
        print_result(result)

    accuracies = np.asarray([result["accuracy"] for result in results], dtype=np.float64)
    print(
        "YDF seed grid accuracy summary: "
        f"mean={accuracies.mean():.4f} "
        f"std={accuracies.std():.4f} "
        f"min={accuracies.min():.4f} "
        f"max={accuracies.max():.4f}"
    )

    output_path = plot_seed_grid(
        results,
        ydf_seed_grid_output,
        "Jovo T7 YDF Sparse Oblique Accuracy vs Pseudorandom Seed",
    )
    print(f"Wrote YDF seed grid plot: {output_path}")
    return results


def do_sporf_projection_grid(
    data_dir,
    train_split,
    ntrees,
    nstreams,
    projection_grid_min,
    projection_grid_max,
    projection_grid_expected_nnz,
    projection_grid_output,
):
    phase("Loading Jovo T7 data")
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

    density = sporf_density_fraction(projection_grid_expected_nnz, x_train.shape[1])
    _, base_sporf_args, _ = base_hyperparameters(
        x_train.shape[1], ntrees, projection_grid_min
    )
    base_sporf_args = base_sporf_args | {
        "density": sporf_density_arg(projection_grid_expected_nnz, x_train.shape[1])
    }

    grid = sporf_projection_grid(projection_grid_min, projection_grid_max)

    results = []
    for num_projections in grid:
        sporf_args = base_sporf_args | {"max_features": int(num_projections)}
        print(
            "Running cuML SPORF projection grid point: "
            f"num_projections={num_projections} "
            f"expected_nnz={projection_grid_expected_nnz:g} "
            f"density={density:.12g}"
        )
        result = train_predict_sporf(
            x_train, y_train, x_test, y_test, sporf_args, nstreams
        )
        result |= {
            "num_projections": int(num_projections),
            "expected_nnz": float(projection_grid_expected_nnz),
            "density": float(density),
        }
        results.append(result)
        print_result(result)

    output_path = plot_sporf_projection_grid(results, projection_grid_output)
    print(f"Wrote projection grid plot: {output_path}")
    return results


def do_synthetic_wide_grid(
    ydf_use_slow_engine,
    output_png,
    output_csv,
    feature_counts,
    n_trials,
    n_train,
    n_test,
    ntrees,
    nstreams,
    num_projections,
    expected_nnz,
    informative_fraction,
    signal_strength,
    base_seed,
):
    feature_grid = list(feature_counts)
    trial_seeds = sporf_seed_grid(base_seed, n_trials)
    prior_quantized_accuracy = read_prior_quantized_accuracy(output_csv)
    rows = []

    for n_features in feature_grid:
        density = sporf_density_fraction(expected_nnz, n_features)
        n_bins = 128
        bootstrap_size_ratio = 0.8
        max_depth = 18
        min_samples_leaf = 2

        sporf_args_base = {
            "max_features": num_projections,
            "max_samples": bootstrap_size_ratio,
            "density": sporf_density_arg(expected_nnz, n_features),
            "n_bins": n_bins,
            "split_criterion": 0,
            "min_samples_leaf": min_samples_leaf,
            "n_estimators": ntrees,
            "max_leaves": -1,
            "max_depth": max_depth,
            "verbose": False,
        }
        ydf_args_base = {
            "label": "foo",
            "bootstrap_size_ratio": bootstrap_size_ratio,
            "max_depth": max_depth,
            "min_examples": min_samples_leaf,
            "num_trees": ntrees,
            "split_axis": "SPARSE_OBLIQUE",
            "sparse_oblique_max_num_projections": num_projections,
            "sparse_oblique_num_projections_exponent": 1.0,
            "sparse_oblique_normalization": "NONE",
            "sparse_oblique_projection_density_factor": expected_nnz,
            "sparse_oblique_weights": "BINARY",
        }
        for trial_idx, seed in enumerate(trial_seeds, start=1):
            print(
                "Synthetic wide trial: "
                f"n_features={n_features} trial={trial_idx}/{n_trials} seed={seed}"
            )
            phase("Synthetic wide: generating data")
            x_train, y_train, x_test, y_test, n_informative = make_synthetic_wide_data(
                n_train=n_train,
                n_test=n_test,
                n_features=n_features,
                informative_fraction=informative_fraction,
                signal_strength=signal_strength,
                seed=seed,
            )
            print(
                f"  n_informative={n_informative} "
                f"density={density:.12g} "
                f"num_projections={num_projections}"
            )

            phase("Synthetic wide: constructing YDF datasets")
            train_ds = make_ydf_dict(x_train, y_train)
            test_ds = make_ydf_dict(x_test)

            sporf_result = train_predict_sporf(
                x_train,
                y_train,
                x_test,
                y_test,
                sporf_args_base | {"random_state": int(seed)},
                nstreams,
            )
            ydf_result = train_predict_ydf_prebuilt(
                "YDF sparse oblique",
                train_ds,
                test_ds,
                y_test,
                ydf_args_base | {"random_seed": int(seed)},
                ydf_use_slow_engine,
            )

            for result in [sporf_result, ydf_result]:
                row = {
                    "model": result["name"],
                    "n_features": n_features,
                    "trial": trial_idx,
                    "seed": int(seed),
                    "n_train": n_train,
                    "n_test": n_test,
                    "n_trees": ntrees,
                    "n_streams": nstreams,
                    "num_projections": num_projections,
                    "expected_nnz": expected_nnz,
                    "max_depth": max_depth,
                    "min_leaf": min_samples_leaf,
                    "n_bins": n_bins,
                    "density": density,
                    "train_time": result["train_time"],
                    "predict_time": result["predict_time"],
                    "accuracy": result["accuracy"],
                }
                rows.append(row)
                print(
                    f"  {row['model']}: train={row['train_time']:.4f}s "
                    f"predict={row['predict_time']:.4f}s "
                    f"accuracy={row['accuracy']:.4f}"
                )

            write_rows_csv(rows, output_csv)
            del train_ds, test_ds, x_train, y_train, x_test, y_test

    csv_path = write_rows_csv(rows, output_csv)
    png_path = plot_synthetic_wide_boxplots(
        rows, output_png, prior_quantized_accuracy
    )
    print(f"Wrote synthetic wide CSV: {csv_path}")
    print(f"Wrote synthetic wide plot: {png_path}")
    return rows


def do_synthetic_wide_plot(output_png, output_csv):
    rows = read_synthetic_wide_csv(output_csv)
    quantized_accuracy = read_prior_quantized_accuracy(output_csv)
    png_path = plot_synthetic_wide_boxplots(rows, output_png, quantized_accuracy)
    print(f"Read synthetic wide CSV: {output_csv}")
    print(f"Wrote synthetic wide plot: {png_path}")
    return png_path


def do_synthetic_friedman_wide_grid(
    ydf_use_slow_engine,
    output_png,
    output_csv,
    feature_counts,
    n_trials,
    n_train,
    n_test,
    ntrees,
    nstreams,
    num_projections,
    expected_nnz,
    informative_fraction,
    noise,
    mode,
    models,
    base_seed,
):
    feature_grid = list(feature_counts)
    trial_seeds = sporf_seed_grid(base_seed, n_trials)
    rows = []
    models = set(models)

    for n_features in feature_grid:
        density = sporf_density_fraction(expected_nnz, n_features)
        n_bins = 128
        bootstrap_size_ratio = 0.8
        max_depth = 18
        min_samples_leaf = 2

        sporf_args_base = {
            "max_features": num_projections,
            "max_samples": bootstrap_size_ratio,
            "density": sporf_density_arg(expected_nnz, n_features),
            "n_bins": n_bins,
            "split_criterion": 2,
            "min_samples_leaf": min_samples_leaf,
            "n_estimators": ntrees,
            "max_leaves": -1,
            "max_depth": max_depth,
            "verbose": False,
        }
        ydf_args_base = {
            "label": "foo",
            "task": ydf.Task.REGRESSION,
            "bootstrap_size_ratio": bootstrap_size_ratio,
            "max_depth": max_depth,
            "min_examples": min_samples_leaf,
            "num_trees": ntrees,
            "split_axis": "SPARSE_OBLIQUE",
            "sparse_oblique_max_num_projections": num_projections,
            "sparse_oblique_num_projections_exponent": 1.0,
            "sparse_oblique_normalization": "NONE",
            "sparse_oblique_projection_density_factor": expected_nnz,
            "sparse_oblique_weights": "BINARY",
        }
        for trial_idx, seed in enumerate(trial_seeds, start=1):
            print(
                "Synthetic Friedman wide regression trial: "
                f"n_features={n_features} trial={trial_idx}/{n_trials} seed={seed}"
            )
            phase("Synthetic Friedman: generating data")
            x_train, y_train, x_test, y_test, n_informative = (
                make_synthetic_friedman_wide_data(
                    n_train=n_train,
                    n_test=n_test,
                    n_features=n_features,
                    informative_fraction=informative_fraction,
                    noise=noise,
                    seed=seed,
                    mode=mode,
                )
            )
            print(
                f"  n_informative={n_informative} mode={mode} "
                f"density={density:.12g} num_projections={num_projections}"
            )

            results = []
            if "sporf" in models:
                results.append(
                    train_predict_sporf_regressor(
                        x_train,
                        y_train,
                        x_test,
                        y_test,
                        sporf_args_base | {"random_state": int(seed)},
                        nstreams,
                    )
                )

            train_ds = None
            test_ds = None
            if "ydf" in models:
                phase("Synthetic Friedman: constructing YDF datasets")
                train_ds = make_ydf_dict(x_train, y_train)
                test_ds = make_ydf_dict(x_test)
                results.append(
                    train_predict_ydf_regressor_prebuilt(
                        "YDF sparse oblique",
                        train_ds,
                        test_ds,
                        y_test,
                        ydf_args_base | {"random_seed": int(seed)},
                        ydf_use_slow_engine,
                    )
                )

            for result in results:
                row = {
                    "model": result["name"],
                    "n_features": n_features,
                    "trial": trial_idx,
                    "seed": int(seed),
                    "n_train": n_train,
                    "n_test": n_test,
                    "n_trees": ntrees,
                    "n_streams": nstreams,
                    "num_projections": num_projections,
                    "expected_nnz": expected_nnz,
                    "max_depth": max_depth,
                    "min_leaf": min_samples_leaf,
                    "n_bins": n_bins,
                    "density": density,
                    "n_informative": n_informative,
                    "informative_fraction": informative_fraction,
                    "friedman_mode": mode,
                    "noise": noise,
                    "train_time": result["train_time"],
                    "predict_time": result["predict_time"],
                    "r2": result["r2"],
                    "rmse": result["rmse"],
                }
                rows.append(row)
                print(
                    f"  {row['model']}: train={row['train_time']:.4f}s "
                    f"predict={row['predict_time']:.4f}s "
                    f"r2={row['r2']:.4f} rmse={row['rmse']:.4f}"
                )

            write_regression_rows_csv(rows, output_csv)
            del train_ds, test_ds, x_train, y_train, x_test, y_test

    csv_path = write_regression_rows_csv(rows, output_csv)
    png_path = plot_synthetic_friedman_wide_boxplots(rows, output_png)
    print(f"Wrote synthetic Friedman wide regression CSV: {csv_path}")
    print(f"Wrote synthetic Friedman wide regression plot: {png_path}")
    return rows


def do_synthetic_friedman_wide_plot(output_png, output_csv):
    rows = read_synthetic_friedman_wide_csv(output_csv)
    png_path = plot_synthetic_friedman_wide_boxplots(rows, output_png)
    print(f"Read synthetic Friedman wide regression CSV: {output_csv}")
    print(f"Wrote synthetic Friedman wide regression plot: {png_path}")
    return png_path


def plot_jovo_wide_boxplots(rows, output_path):
    note = (
        "Training time excludes Jovo data loading, feature subsampling, "
        "and YDF dict construction. Feature subsets are nested prefixes of "
        "seeded random feature permutations."
    )
    model_order = [
        "cuML SPORF",
        "YDF sparse oblique",
        "YDF sparse oblique quantized",
    ]
    models = [
        model
        for model in model_order
        if any(row["model"] == model for row in rows)
    ]
    return plot_wide_boxplots(
        rows=rows,
        output_path=output_path,
        title="Jovo T7 Feature Subsample: SPORF vs YDF Sparse Oblique",
        models=models,
        note=note,
        quality_ceiling=1.01,
    )


def make_jovo_wide_row(
    result,
    n_features,
    trial_idx,
    seed,
    n_train,
    n_test,
    ntrees,
    nstreams,
    num_projections,
    expected_nnz,
    max_depth,
    min_leaf,
    n_bins,
    density,
):
    return {
        "model": result["name"],
        "n_features": n_features,
        "trial": trial_idx,
        "seed": int(seed),
        "n_train": n_train,
        "n_test": n_test,
        "n_trees": ntrees,
        "n_streams": nstreams,
        "num_projections": num_projections,
        "expected_nnz": expected_nnz,
        "max_depth": max_depth,
        "min_leaf": min_leaf,
        "n_bins": n_bins,
        "density": density,
        "train_time": result["train_time"],
        "predict_time": result["predict_time"],
        "accuracy": result["accuracy"],
    }


def do_jovo_wide_grid(
    data_dir,
    train_split,
    ydf_use_slow_engine,
    output_png,
    output_csv,
    feature_counts,
    n_trials,
    ntrees,
    nstreams,
    num_projections,
    expected_nnz,
    models,
    base_seed,
):
    phase("Loading Jovo T7 data")
    X, y, sample_ids, data_args = read_jovo_t7(data_dir)
    print(f"Loaded {data_args}")
    print(f"Target counts: {dict(zip(*np.unique(y, return_counts=True)))}")

    x_train_full, x_test_full, y_train, y_test = train_test_split(
        X, y, train_size=train_split, random_state=123, stratify=y
    )
    x_train_full = np.ascontiguousarray(x_train_full.astype(np.float32, copy=False))
    y_train = np.ascontiguousarray(y_train.astype(np.int32, copy=False))
    x_test_full = np.ascontiguousarray(x_test_full.astype(np.float32, copy=False))
    y_test = np.ascontiguousarray(y_test.astype(np.int32, copy=False))

    feature_grid = list(feature_counts)
    trial_seeds = sporf_seed_grid(base_seed, n_trials)
    rows = []

    n_bins = 128
    bootstrap_size_ratio = 0.8
    max_depth = 18
    min_samples_leaf = 2

    for trial_idx, seed in enumerate(trial_seeds, start=1):
        rng = np.random.default_rng(seed)
        feature_order = rng.permutation(x_train_full.shape[1])

        for n_features in feature_grid:
            density = sporf_density_fraction(expected_nnz, n_features)
            cols = np.sort(feature_order[:n_features])
            print(
                "Jovo wide trial: "
                f"n_features={n_features} trial={trial_idx}/{n_trials} seed={seed}"
            )
            print(
                f"  density={density:.12g} "
                f"num_projections={num_projections}"
            )

            x_train = np.ascontiguousarray(x_train_full[:, cols])
            x_test = np.ascontiguousarray(x_test_full[:, cols])
            train_ds = None
            test_ds = None
            if "ydf" in models or "ydf-quantized" in models:
                phase("Jovo wide: constructing YDF datasets")
                train_ds = make_ydf_dict(x_train, y_train)
                test_ds = make_ydf_dict(x_test)

            sporf_args = {
                "max_features": num_projections,
                "max_samples": bootstrap_size_ratio,
                "density": sporf_density_arg(expected_nnz, n_features),
                "n_bins": n_bins,
                "split_criterion": 0,
                "min_samples_leaf": min_samples_leaf,
                "n_estimators": ntrees,
                "max_leaves": -1,
                "max_depth": max_depth,
                "verbose": False,
                "random_state": int(seed),
            }
            ydf_args = {
                "label": "foo",
                "bootstrap_size_ratio": bootstrap_size_ratio,
                "max_depth": max_depth,
                "min_examples": min_samples_leaf,
                "num_trees": ntrees,
                "split_axis": "SPARSE_OBLIQUE",
                "sparse_oblique_max_num_projections": num_projections,
                "sparse_oblique_num_projections_exponent": 1.0,
                "sparse_oblique_normalization": "NONE",
                "sparse_oblique_projection_density_factor": expected_nnz,
                "sparse_oblique_weights": "BINARY",
                "random_seed": int(seed),
            }
            ydf_quantized_args = ydf_args | {
                "discretize_numerical_columns": True,
                "num_discretized_numerical_bins": n_bins,
                "sorting_strategy": "PRESORT",
            }

            results = []
            if "sporf" in models:
                results.append(
                    train_predict_sporf(
                        x_train, y_train, x_test, y_test, sporf_args, nstreams
                    )
                )
            if "ydf" in models:
                results.append(
                    train_predict_ydf_prebuilt(
                        "YDF sparse oblique",
                        train_ds,
                        test_ds,
                        y_test,
                        ydf_args,
                        ydf_use_slow_engine,
                    )
                )
            if "ydf-quantized" in models:
                results.append(
                    train_predict_ydf_prebuilt(
                        "YDF sparse oblique quantized",
                        train_ds,
                        test_ds,
                        y_test,
                        ydf_quantized_args,
                        ydf_use_slow_engine,
                    )
                )

            if not results:
                raise ValueError("No Jovo feature-scaling models selected")

            for result in results:
                row = make_jovo_wide_row(
                    result,
                    n_features,
                    trial_idx,
                    seed,
                    x_train.shape[0],
                    x_test.shape[0],
                    ntrees,
                    nstreams,
                    num_projections,
                    expected_nnz,
                    max_depth,
                    min_samples_leaf,
                    n_bins,
                    density,
                )
                rows.append(row)
                print(
                    f"  {row['model']}: train={row['train_time']:.4f}s "
                    f"predict={row['predict_time']:.4f}s "
                    f"accuracy={row['accuracy']:.4f}"
                )

            write_rows_csv(rows, output_csv)
            del train_ds, test_ds, x_train, x_test

    csv_path = write_rows_csv(rows, output_csv)
    png_path = plot_jovo_wide_boxplots(rows, output_png)
    print(f"Wrote Jovo wide CSV: {csv_path}")
    print(f"Wrote Jovo wide plot: {png_path}")
    return rows


def do_jovo_wide_plot(output_png, output_csv):
    rows = read_wide_csv(output_csv, include_quantized=True)
    png_path = plot_jovo_wide_boxplots(rows, output_png)
    print(f"Read Jovo wide CSV: {output_csv}")
    print(f"Wrote Jovo wide plot: {png_path}")
    return png_path


def plot_jovo_tree_scale_boxplots(rows, output_path):
    note = (
        "Workstation throughput comparison on Jovo T7. "
        f"cuML uses n_streams=min(n_trees, {DEFAULT_JOVO_TREE_SCALE_NSTREAMS}) "
        "by default; YDF uses default CPU threading. Training time excludes "
        "data loading and YDF dict construction."
    )
    return plot_wide_boxplots(
        rows=rows,
        output_path=output_path,
        title="Jovo T7 Tree Scaling: SPORF vs YDF Sparse Oblique",
        models=[
            "cuML SPORF",
            "YDF sparse oblique",
            "YDF sparse oblique quantized",
        ],
        note=note,
        quality_ceiling=1.01,
        x_field="n_trees",
        x_label="Number of trees",
    )


def make_jovo_tree_scale_row(
    result,
    n_features,
    trial_idx,
    seed,
    n_train,
    n_test,
    ntrees,
    nstreams,
    num_projections,
    expected_nnz,
    max_depth,
    min_leaf,
    n_bins,
    density,
):
    return {
        "model": result["name"],
        "n_features": n_features,
        "trial": trial_idx,
        "seed": int(seed),
        "n_train": n_train,
        "n_test": n_test,
        "n_trees": ntrees,
        "n_streams": nstreams,
        "num_projections": num_projections,
        "expected_nnz": expected_nnz,
        "max_depth": max_depth,
        "min_leaf": min_leaf,
        "n_bins": n_bins,
        "density": density,
        "train_time": result["train_time"],
        "predict_time": result["predict_time"],
        "accuracy": result["accuracy"],
    }


def resolve_tree_scale_nstreams(nstreams, ntrees):
    if nstreams == "auto":
        return ntrees
    return min(int(nstreams), ntrees)


def do_jovo_tree_scale_grid(
    data_dir,
    train_split,
    ydf_use_slow_engine,
    output_png,
    output_csv,
    ntrees_values,
    n_trials,
    nstreams,
    num_projections,
    expected_nnz,
    models,
    n_features,
    base_seed,
):
    phase("Loading Jovo T7 data")
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
    x_train, x_test = subsample_jovo_features(
        x_train, x_test, n_features, base_seed
    )
    if n_features is not None:
        print(f"Subsampled Jovo feature count: {x_train.shape[1]}")

    train_ds = None
    test_ds = None
    if "ydf" in models or "ydf-quantized" in models:
        phase("Jovo tree-scale: constructing YDF datasets")
        train_ds = make_ydf_dict(x_train, y_train)
        test_ds = make_ydf_dict(x_test)

    n_features = x_train.shape[1]
    density = sporf_density_fraction(expected_nnz, n_features)
    n_bins = 128
    bootstrap_size_ratio = 0.8
    max_depth = 18
    min_samples_leaf = 2
    trial_seeds = sporf_seed_grid(base_seed, n_trials)
    rows = []

    for ntrees in ntrees_values:
        resolved_nstreams = resolve_tree_scale_nstreams(nstreams, ntrees)
        for trial_idx, seed in enumerate(trial_seeds, start=1):
            print(
                "Jovo tree-scale trial: "
                f"ntrees={ntrees} trial={trial_idx}/{n_trials} seed={seed}"
            )
            print(
                f"  nstreams={resolved_nstreams} "
                f"density={density:.12g} "
                f"num_projections={num_projections}"
            )

            sporf_args = {
                "max_features": num_projections,
                "max_samples": bootstrap_size_ratio,
                "density": sporf_density_arg(expected_nnz, n_features),
                "n_bins": n_bins,
                "split_criterion": 0,
                "min_samples_leaf": min_samples_leaf,
                "n_estimators": ntrees,
                "max_leaves": -1,
                "max_depth": max_depth,
                "verbose": False,
                "random_state": int(seed),
            }
            ydf_args = {
                "label": "foo",
                "bootstrap_size_ratio": bootstrap_size_ratio,
                "max_depth": max_depth,
                "min_examples": min_samples_leaf,
                "num_trees": ntrees,
                "split_axis": "SPARSE_OBLIQUE",
                "sparse_oblique_max_num_projections": num_projections,
                "sparse_oblique_num_projections_exponent": 1.0,
                "sparse_oblique_normalization": "NONE",
                "sparse_oblique_projection_density_factor": expected_nnz,
                "sparse_oblique_weights": "BINARY",
                "random_seed": int(seed),
                "sorting_strategy": "IN_NODE",
            }
            ydf_quantized_args = ydf_args | {
                "discretize_numerical_columns": True,
                "num_discretized_numerical_bins": n_bins,
                # "sorting_strategy": "PRESORT",
            }

            results = []
            if "sporf" in models:
                results.append(
                    train_predict_sporf(
                        x_train,
                        y_train,
                        x_test,
                        y_test,
                        sporf_args,
                        resolved_nstreams,
                    )
                )
            if "ydf" in models:
                results.append(
                    train_predict_ydf_prebuilt(
                        "YDF sparse oblique",
                        train_ds,
                        test_ds,
                        y_test,
                        ydf_args,
                        ydf_use_slow_engine,
                    )
                )
            if "ydf-quantized" in models:
                results.append(
                    train_predict_ydf_prebuilt(
                        "YDF sparse oblique quantized",
                        train_ds,
                        test_ds,
                        y_test,
                        ydf_quantized_args,
                        ydf_use_slow_engine,
                    )
                )

            if not results:
                raise ValueError("No tree-scale models selected")

            for result in results:
                row = make_jovo_tree_scale_row(
                    result,
                    n_features,
                    trial_idx,
                    seed,
                    x_train.shape[0],
                    x_test.shape[0],
                    ntrees,
                    resolved_nstreams,
                    num_projections,
                    expected_nnz,
                    max_depth,
                    min_samples_leaf,
                    n_bins,
                    density,
                )
                rows.append(row)
                print(
                    f"  {row['model']}: train={row['train_time']:.4f}s "
                    f"predict={row['predict_time']:.4f}s "
                    f"accuracy={row['accuracy']:.4f}"
                )

            write_rows_csv(rows, output_csv)

    csv_path = write_rows_csv(rows, output_csv)
    png_path = plot_jovo_tree_scale_boxplots(rows, output_png)
    print(f"Wrote Jovo tree-scale CSV: {csv_path}")
    print(f"Wrote Jovo tree-scale plot: {png_path}")
    return rows


def do_jovo_tree_scale_plot(output_png, output_csv):
    rows = read_wide_csv(output_csv, include_quantized=True)
    png_path = plot_jovo_tree_scale_boxplots(rows, output_png)
    print(f"Read Jovo tree-scale CSV: {output_csv}")
    print(f"Wrote Jovo tree-scale plot: {png_path}")
    return png_path


def plot_jovo_density_scale_boxplots(rows, output_path):
    note = (
        "Density scaling comparison on Jovo T7. "
        f"cuML uses n_streams=min(n_trees, {DEFAULT_JOVO_TREE_SCALE_NSTREAMS}) "
        "by default; YDF uses default CPU threading. Training time excludes "
        "data loading and YDF dict construction. Density fraction is "
        "E[NNZ] / n_features. YDF quantized omitted from density scouting "
        "after flat low accuracy in the first sweep."
    )
    model_order = [
        "cuML SPORF",
        "YDF sparse oblique",
        "YDF sparse oblique quantized",
    ]
    models = [
        model
        for model in model_order
        if any(row["model"] == model for row in rows)
    ]
    return plot_wide_boxplots(
        rows=rows,
        output_path=output_path,
        title="Jovo T7 Density Scaling: SPORF vs YDF Sparse Oblique",
        models=models,
        note=note,
        quality_ceiling=1.01,
        x_field="expected_nnz",
        x_label="E[NNZ] per projection",
    )


def do_jovo_density_scale_grid(
    data_dir,
    train_split,
    ydf_use_slow_engine,
    output_png,
    output_csv,
    output_hparams_jsonl,
    expected_nnz_values,
    n_trials,
    ntrees,
    nstreams,
    num_projections,
    models,
    n_features,
    base_seed,
):
    phase("Loading Jovo T7 data")
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
    x_train, x_test = subsample_jovo_features(
        x_train, x_test, n_features, base_seed
    )
    if n_features is not None:
        print(f"Subsampled Jovo feature count: {x_train.shape[1]}")

    train_ds = None
    test_ds = None
    if "ydf" in models:
        phase("Jovo density-scale: constructing YDF datasets")
        train_ds = make_ydf_dict(x_train, y_train)
        test_ds = make_ydf_dict(x_test)

    n_features = x_train.shape[1]
    resolved_nstreams = resolve_tree_scale_nstreams(nstreams, ntrees)
    n_bins = 128
    bootstrap_size_ratio = 0.8
    max_depth = 18
    min_samples_leaf = 2
    trial_seeds = sporf_seed_grid(base_seed, n_trials)
    rows = []
    hparam_records = []

    for expected_nnz in expected_nnz_values:
        density = sporf_density_fraction(expected_nnz, n_features)
        for trial_idx, seed in enumerate(trial_seeds, start=1):
            print(
                "Jovo density-scale trial: "
                f"expected_nnz={expected_nnz:g} "
                f"trial={trial_idx}/{n_trials} seed={seed}"
            )
            print(
                f"  ntrees={ntrees} "
                f"nstreams={resolved_nstreams} "
                f"density={density:.12g} "
                f"num_projections={num_projections}"
            )

            sporf_args = {
                "max_features": num_projections,
                "max_samples": bootstrap_size_ratio,
                "density": sporf_density_arg(expected_nnz, n_features),
                "n_bins": n_bins,
                "split_criterion": 0,
                "min_samples_leaf": min_samples_leaf,
                "n_estimators": ntrees,
                "max_leaves": -1,
                "max_depth": max_depth,
                "verbose": False,
                "random_state": int(seed),
            }
            ydf_args = {
                "label": "foo",
                "bootstrap_size_ratio": bootstrap_size_ratio,
                "max_depth": max_depth,
                "min_examples": min_samples_leaf,
                "num_trees": ntrees,
                "split_axis": "SPARSE_OBLIQUE",
                "sparse_oblique_max_num_projections": num_projections,
                "sparse_oblique_num_projections_exponent": 1.0,
                "sparse_oblique_normalization": "NONE",
                "sparse_oblique_projection_density_factor": expected_nnz,
                "sparse_oblique_weights": "BINARY",
                "random_seed": int(seed),
            }

            results = []
            if "sporf" in models:
                results.append(
                    train_predict_sporf(
                        x_train,
                        y_train,
                        x_test,
                        y_test,
                        sporf_args,
                        resolved_nstreams,
                    )
                )
            if "ydf" in models:
                results.append(
                    train_predict_ydf_prebuilt(
                        "YDF sparse oblique",
                        train_ds,
                        test_ds,
                        y_test,
                        ydf_args,
                        ydf_use_slow_engine,
                    )
                )

            if not results:
                raise ValueError("No density-scale models selected")

            for result in results:
                row = make_jovo_tree_scale_row(
                    result,
                    n_features,
                    trial_idx,
                    seed,
                    x_train.shape[0],
                    x_test.shape[0],
                    ntrees,
                    resolved_nstreams,
                    num_projections,
                    expected_nnz,
                    max_depth,
                    min_samples_leaf,
                    n_bins,
                    density,
                )
                rows.append(row)
                hparam_records.append(
                    {
                        "model": row["model"],
                        "n_features": n_features,
                        "trial": trial_idx,
                        "seed": int(seed),
                        "n_train": x_train.shape[0],
                        "n_test": x_test.shape[0],
                        "n_trees": ntrees,
                        "n_streams": resolved_nstreams,
                        "num_projections": num_projections,
                        "expected_nnz": expected_nnz,
                        "density_fraction": density,
                        "hyperparameters": result["hyperparameters"],
                    }
                )
                print(
                    f"  {row['model']}: train={row['train_time']:.4f}s "
                    f"predict={row['predict_time']:.4f}s "
                    f"accuracy={row['accuracy']:.4f}"
                )

            write_rows_csv(rows, output_csv)
            write_commented_jsonl(hparam_records, output_hparams_jsonl)

    csv_path = write_rows_csv(rows, output_csv)
    hparams_path = write_commented_jsonl(hparam_records, output_hparams_jsonl)
    png_path = plot_jovo_density_scale_boxplots(rows, output_png)
    print(f"Wrote Jovo density-scale CSV: {csv_path}")
    print(f"Wrote Jovo density-scale hyperparameters JSONL: {hparams_path}")
    print(f"Wrote Jovo density-scale plot: {png_path}")
    return rows


def do_jovo_density_scale_plot(output_png, output_csv):
    rows = read_wide_csv(output_csv, include_quantized=False)
    png_path = plot_jovo_density_scale_boxplots(rows, output_png)
    print(f"Read Jovo density-scale CSV: {output_csv}")
    print(f"Wrote Jovo density-scale plot: {png_path}")
    return png_path


def do_jovo(
    data_dir,
    train_split,
    ydf_use_slow_engine,
    trial,
    ntrees,
    nstreams,
    max_features,
):
    phase("Loading Jovo T7 data")
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
        "sorting_strategy": "PRESORT",
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
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def apply_shared_arg_overrides(args):
    if args.shared_output_png is not None:
        output_png_attrs = {
            "sporf-density-grid": "density_grid_output",
            "sporf-projection-grid": "projection_grid_output",
            "sporf-seed-grid": "seed_grid_output",
            "ydf-seed-grid": "ydf_seed_grid_output",
            "synthetic-wide-grid": "synthetic_wide_output_png",
            "synthetic-wide-plot": "synthetic_wide_output_png",
            "synthetic-friedman-wide-grid": "synthetic_friedman_wide_output_png",
            "synthetic-friedman-wide-plot": "synthetic_friedman_wide_output_png",
            "jovo-wide-grid": "jovo_wide_output_png",
            "jovo-wide-plot": "jovo_wide_output_png",
            "jovo-tree-scale-grid": "jovo_tree_scale_output_png",
            "jovo-tree-scale-plot": "jovo_tree_scale_output_png",
            "jovo-density-scale-grid": "jovo_density_scale_output_png",
            "jovo-density-scale-plot": "jovo_density_scale_output_png",
        }
        attr = output_png_attrs.get(args.trial)
        if attr is not None:
            setattr(args, attr, args.shared_output_png)
    if args.shared_output_csv is not None:
        output_csv_attrs = {
            "synthetic-wide-grid": "synthetic_wide_output_csv",
            "synthetic-wide-plot": "synthetic_wide_output_csv",
            "synthetic-friedman-wide-grid": "synthetic_friedman_wide_output_csv",
            "synthetic-friedman-wide-plot": "synthetic_friedman_wide_output_csv",
            "jovo-wide-grid": "jovo_wide_output_csv",
            "jovo-wide-plot": "jovo_wide_output_csv",
            "jovo-tree-scale-grid": "jovo_tree_scale_output_csv",
            "jovo-tree-scale-plot": "jovo_tree_scale_output_csv",
            "jovo-density-scale-grid": "jovo_density_scale_output_csv",
            "jovo-density-scale-plot": "jovo_density_scale_output_csv",
        }
        attr = output_csv_attrs.get(args.trial)
        if attr is not None:
            setattr(args, attr, args.shared_output_csv)
    if args.shared_output_hparams_jsonl is not None:
        args.jovo_density_scale_output_hparams_jsonl = (
            args.shared_output_hparams_jsonl
        )
    if args.shared_trials is not None:
        args.seed_grid_count = args.shared_trials
        args.synthetic_wide_trials = args.shared_trials
        args.synthetic_friedman_wide_trials = args.shared_trials
        args.jovo_wide_trials = args.shared_trials
        args.jovo_tree_scale_trials = args.shared_trials
        args.jovo_density_scale_trials = args.shared_trials
    if args.shared_base_seed is not None:
        args.seed_grid_base_seed = args.shared_base_seed
        args.synthetic_wide_base_seed = args.shared_base_seed
        args.synthetic_friedman_wide_base_seed = args.shared_base_seed
        args.jovo_wide_base_seed = args.shared_base_seed
        args.jovo_tree_scale_base_seed = args.shared_base_seed
        args.jovo_density_scale_base_seed = args.shared_base_seed
    if args.shared_models is not None:
        model_attrs = {
            "synthetic-friedman-wide-grid": (
                "synthetic_friedman_wide_models",
                {"sporf", "ydf"},
            ),
            "jovo-wide-grid": (
                "jovo_wide_models",
                {"sporf", "ydf", "ydf-quantized"},
            ),
            "jovo-tree-scale-grid": (
                "jovo_tree_scale_models",
                {"sporf", "ydf", "ydf-quantized"},
            ),
            "jovo-density-scale-grid": (
                "jovo_density_scale_models",
                {"sporf", "ydf"},
            ),
        }
        if args.trial in model_attrs:
            attr, allowed = model_attrs[args.trial]
            requested = set(args.shared_models)
            unknown = sorted(requested - allowed)
            if unknown:
                raise ValueError(
                    f"--models contains invalid value(s) for {args.trial}: "
                    + ", ".join(unknown)
                )
            setattr(args, attr, sorted(requested))

    if args.shared_n_train is not None:
        args.synthetic_wide_n_train = args.shared_n_train
        args.synthetic_friedman_wide_n_train = args.shared_n_train
    if args.shared_n_test is not None:
        args.synthetic_wide_n_test = args.shared_n_test
        args.synthetic_friedman_wide_n_test = args.shared_n_test
    if args.nstreams is not None:
        args.synthetic_wide_nstreams = args.nstreams
        args.synthetic_friedman_wide_nstreams = args.nstreams
        args.jovo_wide_nstreams = args.nstreams
        args.jovo_tree_scale_nstreams = str(args.nstreams)
        args.jovo_density_scale_nstreams = str(args.nstreams)
    if args.shared_ntrees is not None:
        args.ntrees = args.shared_ntrees
        args.grid_ntrees = args.shared_ntrees
        args.synthetic_wide_ntrees = args.shared_ntrees
        args.synthetic_friedman_wide_ntrees = args.shared_ntrees
        args.jovo_wide_ntrees = args.shared_ntrees
        args.jovo_tree_scale_ntrees = [args.shared_ntrees]
        args.jovo_density_scale_ntrees = args.shared_ntrees
    if args.shared_num_projections is not None:
        args.density_grid_num_projections = args.shared_num_projections
        args.seed_grid_num_projections = args.shared_num_projections
        args.synthetic_wide_num_projections = args.shared_num_projections
        args.synthetic_friedman_wide_num_projections = args.shared_num_projections
        args.jovo_wide_num_projections = args.shared_num_projections
        args.jovo_tree_scale_num_projections = args.shared_num_projections
        args.jovo_density_scale_num_projections = args.shared_num_projections
    if args.shared_expected_nnz is not None:
        first_expected_nnz = args.shared_expected_nnz[0]
        args.projection_grid_expected_nnz = first_expected_nnz
        args.seed_grid_expected_nnz = first_expected_nnz
        args.synthetic_wide_expected_nnz = first_expected_nnz
        args.synthetic_friedman_wide_expected_nnz = first_expected_nnz
        args.jovo_wide_expected_nnz = first_expected_nnz
        args.jovo_tree_scale_expected_nnz = first_expected_nnz
        args.jovo_density_scale_expected_nnz = args.shared_expected_nnz
    if args.shared_n_features is not None:
        if args.trial in ("synthetic-wide-grid", "synthetic-wide-plot"):
            args.synthetic_wide_feature_counts = args.shared_n_features
        elif args.trial in (
            "synthetic-friedman-wide-grid",
            "synthetic-friedman-wide-plot",
        ):
            args.synthetic_friedman_wide_feature_counts = args.shared_n_features
        elif args.trial in ("jovo-wide-grid", "jovo-wide-plot"):
            args.jovo_wide_feature_counts = args.shared_n_features
        elif args.trial in ("jovo-tree-scale-grid", "jovo-density-scale-grid"):
            if len(args.shared_n_features) != 1:
                raise ValueError(
                    "--n-features accepts one value for Jovo tree/density scaling"
                )
            args.jovo_scale_n_features = args.shared_n_features[0]


def parse_args():
    parser = argparse.ArgumentParser(usage="%(prog)s [options]")
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
        choices=[
            "all",
            "sporf",
            "sporf-density-grid",
            "sporf-projection-grid",
            "sporf-seed-grid",
            "ydf-seed-grid",
            "synthetic-wide-grid",
            "synthetic-wide-plot",
            "synthetic-friedman-wide-grid",
            "synthetic-friedman-wide-plot",
            "jovo-wide-grid",
            "jovo-wide-plot",
            "jovo-tree-scale-grid",
            "jovo-tree-scale-plot",
            "jovo-density-scale-grid",
            "jovo-density-scale-plot",
        ],
        default="all",
        help=(
            "Trial to run: full comparison, cuML SPORF only, SPORF density grid, "
            "SPORF projection grid, SPORF seed grid, YDF seed grid, or synthetic "
            "wide/Jovo feature-subsample comparison grid/plot."
        ),
    )
    parser.add_argument(
        "--ntrees",
        default=4,
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--grid-ntrees",
        default=DEFAULT_GRID_NTREES,
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--n-streams",
        dest="nstreams",
        default=None,
        type=int,
        help=(
            "Number of cuML streams. Defaults to --ntrees for normal trials "
            f"and {DEFAULT_GRID_NSTREAMS} for grid trials."
        ),
    )
    parser.add_argument(
        "--n-train",
        dest="shared_n_train",
        default=None,
        type=int,
        help="Synthetic training row count override.",
    )
    parser.add_argument(
        "--n-test",
        dest="shared_n_test",
        default=None,
        type=int,
        help="Synthetic test row count override.",
    )
    parser.add_argument(
        "--n-trees",
        dest="shared_ntrees",
        default=None,
        type=int,
        help="Tree-count override for non-tree-sweep grids.",
    )
    parser.add_argument(
        "--n-features",
        dest="shared_n_features",
        default=None,
        type=int,
        nargs="+",
        help=(
            "Feature-count override. Accepts multiple values for feature "
            "scaling grids and one value for Jovo tree/density scaling."
        ),
    )
    parser.add_argument(
        "--expected_nnz",
        dest="shared_expected_nnz",
        default=None,
        type=float,
        nargs="+",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--expected-nnz",
        dest="shared_expected_nnz",
        default=None,
        type=float,
        nargs="+",
        help=(
            "E[NNZ] override. Multiple values define the density-scaling "
            "grid; other trials use the first value."
        ),
    )
    parser.add_argument(
        "--num_projections",
        dest="shared_num_projections",
        default=None,
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--num-projections",
        dest="shared_num_projections",
        default=None,
        type=int,
        help="Sparse projection count per node override.",
    )
    parser.add_argument(
        "--output-png",
        dest="shared_output_png",
        default=None,
        type=Path,
        help="PNG output path for the selected grid or plot trial.",
    )
    parser.add_argument(
        "--output-csv",
        dest="shared_output_csv",
        default=None,
        type=Path,
        help="CSV input/output path for the selected grid or plot trial.",
    )
    parser.add_argument(
        "--output-hparams-jsonl",
        dest="shared_output_hparams_jsonl",
        default=None,
        type=Path,
        help="Density-scale hyperparameter JSONL sidecar output path.",
    )
    parser.add_argument(
        "--trials",
        dest="shared_trials",
        default=None,
        type=int,
        help="Number of seed trials per grid point where applicable.",
    )
    parser.add_argument(
        "--base-seed",
        dest="shared_base_seed",
        default=None,
        type=int,
        help="Base seed for trials that generate per-run seeds.",
    )
    parser.add_argument(
        "--models",
        dest="shared_models",
        choices=["sporf", "ydf", "ydf-quantized"],
        nargs="+",
        default=None,
        help="Models to run for model-selectable grid trials.",
    )
    parser.add_argument(
        "--max-features",
        default=DEFAULT_MAX_FEATURES,
        type=parse_max_features,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--density-grid-steps",
        default=8,
        type=int,
        help="Number of log-spaced SPORF density grid points.",
    )
    parser.add_argument(
        "--density-grid-min-expected-nnz",
        default=1.0,
        type=float,
        help="Minimum density grid point as absolute E[NNZ].",
    )
    parser.add_argument(
        "--density-grid-max-density",
        default=0.5,
        type=float,
        help="Maximum density grid point as a density fraction.",
    )
    parser.add_argument(
        "--density-grid-num-projections",
        default=10,
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--density-grid-output",
        default=Path("jovo_sporf_density_grid.png"),
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--projection-grid-min",
        default=1,
        type=int,
        help="Minimum projections per node for the SPORF projection grid.",
    )
    parser.add_argument(
        "--projection-grid-max",
        default=128,
        type=int,
        help="Maximum projections per node for the SPORF projection grid.",
    )
    parser.add_argument(
        "--projection-grid-expected-nnz",
        default=2.0,
        type=float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--projection-grid-output",
        default=Path("jovo_sporf_projection_grid.png"),
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--seed-grid-base-seed",
        default=12345,
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--seed-grid-count",
        default=10,
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--seed-grid-expected-nnz",
        default=2.0,
        type=float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--seed-grid-num-projections",
        default=10,
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--seed-grid-output",
        default=Path("jovo_sporf_seed_grid.png"),
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ydf-seed-grid-output",
        default=Path("jovo_ydf_seed_grid.png"),
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--synthetic-wide-output-png",
        default=Path("synthetic_sporf_ydf_wide_grid.png"),
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--synthetic-wide-output-csv",
        default=Path("synthetic_sporf_ydf_wide_grid.csv"),
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--synthetic-wide-feature-counts",
        default=DEFAULT_SYNTHETIC_WIDE_FEATURES,
        type=int,
        nargs="+",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--synthetic-wide-min-features",
        default=100,
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--synthetic-wide-max-features",
        default=500_000,
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--synthetic-wide-feature-steps",
        default=8,
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--synthetic-wide-trials",
        default=10,
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--synthetic-wide-n-train",
        default=800,
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--synthetic-wide-n-test",
        default=200,
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--synthetic-wide-ntrees",
        default=DEFAULT_SYNTHETIC_WIDE_NTREES,
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--synthetic-wide-nstreams",
        default=DEFAULT_SYNTHETIC_WIDE_NSTREAMS,
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--synthetic-wide-num-projections",
        default=10,
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--synthetic-wide-expected-nnz",
        default=2.0,
        type=float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--synthetic-wide-informative-fraction",
        default=0.5,
        type=float,
        help="Fraction of synthetic features with class-dependent signal.",
    )
    parser.add_argument(
        "--synthetic-wide-signal-strength",
        default=0.5,
        type=float,
        help="Mean shift applied to informative features by class.",
    )
    parser.add_argument(
        "--synthetic-wide-base-seed",
        default=20260612,
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--synthetic-friedman-wide-output-png",
        default=Path("synthetic_friedman_sporf_ydf_wide_grid.png"),
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--synthetic-friedman-wide-output-csv",
        default=Path("synthetic_friedman_sporf_ydf_wide_grid.csv"),
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--synthetic-friedman-wide-feature-counts",
        default=DEFAULT_SYNTHETIC_WIDE_FEATURES,
        type=int,
        nargs="+",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--synthetic-friedman-wide-trials",
        default=10,
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--synthetic-friedman-wide-n-train",
        default=800,
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--synthetic-friedman-wide-n-test",
        default=200,
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--synthetic-friedman-wide-ntrees",
        default=DEFAULT_SYNTHETIC_FRIEDMAN_WIDE_NTREES,
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--synthetic-friedman-wide-nstreams",
        default=DEFAULT_SYNTHETIC_FRIEDMAN_WIDE_NSTREAMS,
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--synthetic-friedman-wide-num-projections",
        default=10,
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--synthetic-friedman-wide-expected-nnz",
        default=2.0,
        type=float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--synthetic-friedman-wide-informative-fraction",
        default=0.5,
        type=float,
        help=(
            "Fraction of synthetic features used by additive Friedman blocks "
            "when --synthetic-friedman-wide-mode=blocks."
        ),
    )
    parser.add_argument(
        "--synthetic-friedman-wide-noise",
        default=1.0,
        type=float,
        help="Gaussian noise standard deviation for synthetic Friedman targets.",
    )
    parser.add_argument(
        "--synthetic-friedman-wide-mode",
        choices=["blocks", "canonical"],
        default="blocks",
        help=(
            "Regression target: additive Friedman1 blocks over the informative "
            "fraction, or canonical sklearn make_friedman1 with five informative "
            "features."
        ),
    )
    parser.add_argument(
        "--synthetic-friedman-wide-models",
        choices=["sporf", "ydf"],
        nargs="+",
        default=["sporf", "ydf"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--synthetic-friedman-wide-base-seed",
        default=20260630,
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--jovo-wide-output-png",
        default=Path("jovo_sporf_ydf_wide_grid.png"),
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--jovo-wide-output-csv",
        default=Path("jovo_sporf_ydf_wide_grid.csv"),
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--jovo-wide-feature-counts",
        default=DEFAULT_JOVO_WIDE_FEATURES,
        type=int,
        nargs="+",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--jovo-wide-trials",
        "--jovo-wide-ntrials",
        dest="jovo_wide_trials",
        default=10,
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--jovo-wide-ntrees",
        default=DEFAULT_SYNTHETIC_WIDE_NTREES,
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--jovo-wide-nstreams",
        default=DEFAULT_SYNTHETIC_WIDE_NSTREAMS,
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--jovo-wide-num-projections",
        default=10,
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--jovo-wide-expected-nnz",
        default=2.0,
        type=float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--jovo-wide-models",
        choices=["sporf", "ydf", "ydf-quantized"],
        default=["sporf", "ydf", "ydf-quantized"],
        nargs="+",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--jovo-wide-base-seed",
        default=20260613,
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--jovo-tree-scale-output-png",
        default=Path("jovo_sporf_ydf_tree_scale.png"),
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--jovo-tree-scale-output-csv",
        default=Path("jovo_sporf_ydf_tree_scale.csv"),
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--jovo-tree-scale-ntrees",
        default=DEFAULT_JOVO_TREE_SCALE_NTREES,
        type=int,
        nargs="+",
        help="Tree counts for the Jovo tree-scaling grid.",
    )
    parser.add_argument(
        "--jovo-tree-scale-trials",
        default=3,
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--jovo-tree-scale-nstreams",
        default=str(DEFAULT_JOVO_TREE_SCALE_NSTREAMS),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--jovo-tree-scale-num-projections",
        default=10,
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--jovo-tree-scale-expected-nnz",
        default=2.0,
        type=float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--jovo-tree-scale-models",
        choices=["sporf", "ydf", "ydf-quantized"],
        default=["sporf", "ydf", "ydf-quantized"],
        nargs="+",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--jovo-tree-scale-base-seed",
        default=20260615,
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--jovo-density-scale-output-png",
        default=Path("jovo_sporf_ydf_density_scale.png"),
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--jovo-density-scale-output-csv",
        default=Path("jovo_sporf_ydf_density_scale.csv"),
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--jovo-density-scale-output-hparams-jsonl",
        default=None,
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--jovo-density-scale-expected-nnz",
        default=DEFAULT_JOVO_DENSITY_SCALE_EXPECTED_NNZ,
        type=float,
        nargs="+",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--jovo-density-scale-trials",
        default=1,
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--jovo-density-scale-ntrees",
        default=4,
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--jovo-density-scale-nstreams",
        default=str(DEFAULT_JOVO_TREE_SCALE_NSTREAMS),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--jovo-density-scale-num-projections",
        default=10,
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--jovo-density-scale-models",
        choices=["sporf", "ydf"],
        default=["sporf", "ydf"],
        nargs="+",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--jovo-density-scale-base-seed",
        default=20260616,
        type=int,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    args.jovo_scale_n_features = None
    try:
        apply_shared_arg_overrides(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.jovo_density_scale_output_hparams_jsonl is None:
        args.jovo_density_scale_output_hparams_jsonl = (
            args.jovo_density_scale_output_csv.with_suffix(".hparams.jsonl")
        )
    if args.ntrees < 1:
        parser.error("--ntrees must be at least 1")
    if args.grid_ntrees < 1:
        parser.error("--grid-ntrees must be at least 1")
    if args.nstreams is not None and args.nstreams < 1:
        parser.error("--n-streams must be at least 1")
    if args.density_grid_steps < 2:
        parser.error("--density-grid-steps must be at least 2")
    if args.density_grid_min_expected_nnz <= 0:
        parser.error("--density-grid-min-expected-nnz must be positive")
    if not 0 < args.density_grid_max_density <= 1:
        parser.error("--density-grid-max-density must be in (0, 1]")
    if args.density_grid_num_projections < 1:
        parser.error("--density-grid-num-projections must be at least 1")
    if args.projection_grid_min < 1:
        parser.error("--projection-grid-min must be at least 1")
    if args.projection_grid_max < args.projection_grid_min:
        parser.error("--projection-grid-max must be at least --projection-grid-min")
    if args.projection_grid_expected_nnz <= 0:
        parser.error("--projection-grid-expected-nnz must be positive")
    if args.seed_grid_count < 1:
        parser.error("--seed-grid-count must be at least 1")
    if args.seed_grid_expected_nnz <= 0:
        parser.error("--seed-grid-expected-nnz must be positive")
    if args.seed_grid_num_projections < 1:
        parser.error("--seed-grid-num-projections must be at least 1")
    if not args.synthetic_wide_feature_counts:
        parser.error("--synthetic-wide-feature-counts cannot be empty")
    if any(value < 1 for value in args.synthetic_wide_feature_counts):
        parser.error("--synthetic-wide-feature-counts values must be at least 1")
    args.synthetic_wide_feature_counts = sorted(set(args.synthetic_wide_feature_counts))
    if args.synthetic_wide_trials < 1:
        parser.error("--synthetic-wide-trials must be at least 1")
    if args.synthetic_wide_n_train < 1:
        parser.error("--synthetic-wide-n-train must be at least 1")
    if args.synthetic_wide_n_test < 1:
        parser.error("--synthetic-wide-n-test must be at least 1")
    if args.synthetic_wide_ntrees < 1:
        parser.error("--synthetic-wide-ntrees must be at least 1")
    if args.synthetic_wide_nstreams < 1:
        parser.error("--synthetic-wide-nstreams must be at least 1")
    if args.synthetic_wide_num_projections < 1:
        parser.error("--synthetic-wide-num-projections must be at least 1")
    if args.synthetic_wide_expected_nnz <= 0:
        parser.error("--synthetic-wide-expected-nnz must be positive")
    if args.synthetic_wide_expected_nnz > min(args.synthetic_wide_feature_counts):
        parser.error(
            "--synthetic-wide-expected-nnz cannot exceed "
            "the minimum --synthetic-wide-feature-counts value"
        )
    if not 0 < args.synthetic_wide_informative_fraction <= 1:
        parser.error("--synthetic-wide-informative-fraction must be in (0, 1]")
    if args.synthetic_wide_signal_strength <= 0:
        parser.error("--synthetic-wide-signal-strength must be positive")
    if not args.synthetic_friedman_wide_feature_counts:
        parser.error("--synthetic-friedman-wide-feature-counts cannot be empty")
    if any(value < 5 for value in args.synthetic_friedman_wide_feature_counts):
        parser.error("--synthetic-friedman-wide-feature-counts values must be at least 5")
    args.synthetic_friedman_wide_feature_counts = sorted(
        set(args.synthetic_friedman_wide_feature_counts)
    )
    if args.synthetic_friedman_wide_trials < 1:
        parser.error("--synthetic-friedman-wide-trials must be at least 1")
    if args.synthetic_friedman_wide_n_train < 1:
        parser.error("--synthetic-friedman-wide-n-train must be at least 1")
    if args.synthetic_friedman_wide_n_test < 1:
        parser.error("--synthetic-friedman-wide-n-test must be at least 1")
    if args.synthetic_friedman_wide_ntrees < 1:
        parser.error("--synthetic-friedman-wide-ntrees must be at least 1")
    if args.synthetic_friedman_wide_nstreams < 1:
        parser.error("--synthetic-friedman-wide-nstreams must be at least 1")
    if args.synthetic_friedman_wide_num_projections < 1:
        parser.error("--synthetic-friedman-wide-num-projections must be at least 1")
    if args.synthetic_friedman_wide_expected_nnz <= 0:
        parser.error("--synthetic-friedman-wide-expected-nnz must be positive")
    if args.synthetic_friedman_wide_expected_nnz > min(
        args.synthetic_friedman_wide_feature_counts
    ):
        parser.error(
            "--synthetic-friedman-wide-expected-nnz cannot exceed "
            "the minimum --synthetic-friedman-wide-feature-counts value"
        )
    if not 0 < args.synthetic_friedman_wide_informative_fraction <= 1:
        parser.error("--synthetic-friedman-wide-informative-fraction must be in (0, 1]")
    if args.synthetic_friedman_wide_noise < 0:
        parser.error("--synthetic-friedman-wide-noise cannot be negative")
    args.synthetic_friedman_wide_models = sorted(
        set(args.synthetic_friedman_wide_models)
    )
    if not args.jovo_wide_feature_counts:
        parser.error("--jovo-wide-feature-counts cannot be empty")
    if any(value < 1 for value in args.jovo_wide_feature_counts):
        parser.error("--jovo-wide-feature-counts values must be at least 1")
    if any(value > JOVO_T7_N_FEATURES for value in args.jovo_wide_feature_counts):
        parser.error(
            f"--jovo-wide-feature-counts cannot exceed {JOVO_T7_N_FEATURES}"
        )
    args.jovo_wide_feature_counts = sorted(set(args.jovo_wide_feature_counts))
    if args.jovo_wide_trials < 1:
        parser.error("--jovo-wide-trials must be at least 1")
    if args.jovo_wide_ntrees < 1:
        parser.error("--jovo-wide-ntrees must be at least 1")
    if args.jovo_wide_nstreams < 1:
        parser.error("--jovo-wide-nstreams must be at least 1")
    if args.jovo_wide_num_projections < 1:
        parser.error("--jovo-wide-num-projections must be at least 1")
    if args.jovo_wide_expected_nnz <= 0:
        parser.error("--jovo-wide-expected-nnz must be positive")
    if args.jovo_wide_expected_nnz > min(args.jovo_wide_feature_counts):
        parser.error(
            "--jovo-wide-expected-nnz cannot exceed "
            "the minimum --jovo-wide-feature-counts value"
        )
    args.jovo_wide_models = sorted(set(args.jovo_wide_models))
    if not args.jovo_tree_scale_ntrees:
        parser.error("--jovo-tree-scale-ntrees cannot be empty")
    if any(value < 1 for value in args.jovo_tree_scale_ntrees):
        parser.error("--jovo-tree-scale-ntrees values must be at least 1")
    args.jovo_tree_scale_ntrees = sorted(set(args.jovo_tree_scale_ntrees))
    if args.jovo_tree_scale_trials < 1:
        parser.error("--jovo-tree-scale-trials must be at least 1")
    if args.jovo_tree_scale_nstreams != "auto":
        try:
            args.jovo_tree_scale_nstreams = int(args.jovo_tree_scale_nstreams)
        except ValueError:
            parser.error("--jovo-tree-scale-nstreams must be 'auto' or an integer")
        if args.jovo_tree_scale_nstreams < 1:
            parser.error("--jovo-tree-scale-nstreams must be at least 1")
    if args.jovo_tree_scale_num_projections < 1:
        parser.error("--jovo-tree-scale-num-projections must be at least 1")
    if args.jovo_tree_scale_expected_nnz <= 0:
        parser.error("--jovo-tree-scale-expected-nnz must be positive")
    if args.jovo_tree_scale_expected_nnz > JOVO_T7_N_FEATURES:
        parser.error(
            f"--jovo-tree-scale-expected-nnz cannot exceed {JOVO_T7_N_FEATURES}"
        )
    args.jovo_tree_scale_models = sorted(set(args.jovo_tree_scale_models))
    if not args.jovo_density_scale_expected_nnz:
        parser.error("--jovo-density-scale-expected-nnz cannot be empty")
    if any(value <= 0 for value in args.jovo_density_scale_expected_nnz):
        parser.error("--jovo-density-scale-expected-nnz values must be positive")
    if any(value > JOVO_T7_N_FEATURES for value in args.jovo_density_scale_expected_nnz):
        parser.error(
            f"--jovo-density-scale-expected-nnz cannot exceed {JOVO_T7_N_FEATURES}"
        )
    args.jovo_density_scale_expected_nnz = sorted(
        {float(value) for value in args.jovo_density_scale_expected_nnz}
    )
    if args.jovo_density_scale_trials < 1:
        parser.error("--jovo-density-scale-trials must be at least 1")
    if args.jovo_density_scale_ntrees < 1:
        parser.error("--jovo-density-scale-ntrees must be at least 1")
    if args.jovo_density_scale_nstreams != "auto":
        try:
            args.jovo_density_scale_nstreams = int(args.jovo_density_scale_nstreams)
        except ValueError:
            parser.error("--jovo-density-scale-nstreams must be 'auto' or an integer")
        if args.jovo_density_scale_nstreams < 1:
            parser.error("--jovo-density-scale-nstreams must be at least 1")
    if args.jovo_density_scale_num_projections < 1:
        parser.error("--jovo-density-scale-num-projections must be at least 1")
    args.jovo_density_scale_models = sorted(set(args.jovo_density_scale_models))
    return args


def main():
    args = parse_args()
    if args.trial == "sporf-density-grid":
        nstreams = args.nstreams if args.nstreams is not None else DEFAULT_GRID_NSTREAMS
        do_sporf_density_grid(
            data_dir=args.data_dir,
            train_split=args.train_split,
            ntrees=args.grid_ntrees,
            nstreams=nstreams,
            density_grid_steps=args.density_grid_steps,
            density_grid_min_expected_nnz=args.density_grid_min_expected_nnz,
            density_grid_max_density=args.density_grid_max_density,
            density_grid_num_projections=args.density_grid_num_projections,
            density_grid_output=args.density_grid_output,
        )
        return

    if args.trial == "sporf-projection-grid":
        nstreams = args.nstreams if args.nstreams is not None else DEFAULT_GRID_NSTREAMS
        do_sporf_projection_grid(
            data_dir=args.data_dir,
            train_split=args.train_split,
            ntrees=args.grid_ntrees,
            nstreams=nstreams,
            projection_grid_min=args.projection_grid_min,
            projection_grid_max=args.projection_grid_max,
            projection_grid_expected_nnz=args.projection_grid_expected_nnz,
            projection_grid_output=args.projection_grid_output,
        )
        return

    if args.trial == "sporf-seed-grid":
        nstreams = args.nstreams if args.nstreams is not None else DEFAULT_GRID_NSTREAMS
        do_sporf_seed_grid(
            data_dir=args.data_dir,
            train_split=args.train_split,
            ntrees=args.grid_ntrees,
            nstreams=nstreams,
            seed_grid_base_seed=args.seed_grid_base_seed,
            seed_grid_count=args.seed_grid_count,
            seed_grid_expected_nnz=args.seed_grid_expected_nnz,
            seed_grid_num_projections=args.seed_grid_num_projections,
            seed_grid_output=args.seed_grid_output,
        )
        return

    if args.trial == "ydf-seed-grid":
        do_ydf_seed_grid(
            data_dir=args.data_dir,
            train_split=args.train_split,
            ydf_use_slow_engine=not args.ydf_fast_engine,
            ntrees=args.grid_ntrees,
            seed_grid_base_seed=args.seed_grid_base_seed,
            seed_grid_count=args.seed_grid_count,
            seed_grid_expected_nnz=args.seed_grid_expected_nnz,
            seed_grid_num_projections=args.seed_grid_num_projections,
            ydf_seed_grid_output=args.ydf_seed_grid_output,
        )
        return

    if args.trial == "synthetic-wide-grid":
        do_synthetic_wide_grid(
            ydf_use_slow_engine=not args.ydf_fast_engine,
            output_png=args.synthetic_wide_output_png,
            output_csv=args.synthetic_wide_output_csv,
            feature_counts=args.synthetic_wide_feature_counts,
            n_trials=args.synthetic_wide_trials,
            n_train=args.synthetic_wide_n_train,
            n_test=args.synthetic_wide_n_test,
            ntrees=args.synthetic_wide_ntrees,
            nstreams=args.synthetic_wide_nstreams,
            num_projections=args.synthetic_wide_num_projections,
            expected_nnz=args.synthetic_wide_expected_nnz,
            informative_fraction=args.synthetic_wide_informative_fraction,
            signal_strength=args.synthetic_wide_signal_strength,
            base_seed=args.synthetic_wide_base_seed,
        )
        return

    if args.trial == "synthetic-wide-plot":
        do_synthetic_wide_plot(
            output_png=args.synthetic_wide_output_png,
            output_csv=args.synthetic_wide_output_csv,
        )
        return

    if args.trial == "synthetic-friedman-wide-grid":
        do_synthetic_friedman_wide_grid(
            ydf_use_slow_engine=not args.ydf_fast_engine,
            output_png=args.synthetic_friedman_wide_output_png,
            output_csv=args.synthetic_friedman_wide_output_csv,
            feature_counts=args.synthetic_friedman_wide_feature_counts,
            n_trials=args.synthetic_friedman_wide_trials,
            n_train=args.synthetic_friedman_wide_n_train,
            n_test=args.synthetic_friedman_wide_n_test,
            ntrees=args.synthetic_friedman_wide_ntrees,
            nstreams=args.synthetic_friedman_wide_nstreams,
            num_projections=args.synthetic_friedman_wide_num_projections,
            expected_nnz=args.synthetic_friedman_wide_expected_nnz,
            informative_fraction=args.synthetic_friedman_wide_informative_fraction,
            noise=args.synthetic_friedman_wide_noise,
            mode=args.synthetic_friedman_wide_mode,
            models=args.synthetic_friedman_wide_models,
            base_seed=args.synthetic_friedman_wide_base_seed,
        )
        return

    if args.trial == "synthetic-friedman-wide-plot":
        do_synthetic_friedman_wide_plot(
            output_png=args.synthetic_friedman_wide_output_png,
            output_csv=args.synthetic_friedman_wide_output_csv,
        )
        return

    if args.trial == "jovo-wide-grid":
        do_jovo_wide_grid(
            data_dir=args.data_dir,
            train_split=args.train_split,
            ydf_use_slow_engine=not args.ydf_fast_engine,
            output_png=args.jovo_wide_output_png,
            output_csv=args.jovo_wide_output_csv,
            feature_counts=args.jovo_wide_feature_counts,
            n_trials=args.jovo_wide_trials,
            ntrees=args.jovo_wide_ntrees,
            nstreams=args.jovo_wide_nstreams,
            num_projections=args.jovo_wide_num_projections,
            expected_nnz=args.jovo_wide_expected_nnz,
            models=args.jovo_wide_models,
            base_seed=args.jovo_wide_base_seed,
        )
        return

    if args.trial == "jovo-wide-plot":
        do_jovo_wide_plot(
            output_png=args.jovo_wide_output_png,
            output_csv=args.jovo_wide_output_csv,
        )
        return

    if args.trial == "jovo-tree-scale-grid":
        do_jovo_tree_scale_grid(
            data_dir=args.data_dir,
            train_split=args.train_split,
            ydf_use_slow_engine=not args.ydf_fast_engine,
            output_png=args.jovo_tree_scale_output_png,
            output_csv=args.jovo_tree_scale_output_csv,
            ntrees_values=args.jovo_tree_scale_ntrees,
            n_trials=args.jovo_tree_scale_trials,
            nstreams=args.jovo_tree_scale_nstreams,
            num_projections=args.jovo_tree_scale_num_projections,
            expected_nnz=args.jovo_tree_scale_expected_nnz,
            models=args.jovo_tree_scale_models,
            n_features=args.jovo_scale_n_features,
            base_seed=args.jovo_tree_scale_base_seed,
        )
        return

    if args.trial == "jovo-tree-scale-plot":
        do_jovo_tree_scale_plot(
            output_png=args.jovo_tree_scale_output_png,
            output_csv=args.jovo_tree_scale_output_csv,
        )
        return

    if args.trial == "jovo-density-scale-grid":
        do_jovo_density_scale_grid(
            data_dir=args.data_dir,
            train_split=args.train_split,
            ydf_use_slow_engine=not args.ydf_fast_engine,
            output_png=args.jovo_density_scale_output_png,
            output_csv=args.jovo_density_scale_output_csv,
            output_hparams_jsonl=args.jovo_density_scale_output_hparams_jsonl,
            expected_nnz_values=args.jovo_density_scale_expected_nnz,
            n_trials=args.jovo_density_scale_trials,
            ntrees=args.jovo_density_scale_ntrees,
            nstreams=args.jovo_density_scale_nstreams,
            num_projections=args.jovo_density_scale_num_projections,
            models=args.jovo_density_scale_models,
            n_features=args.jovo_scale_n_features,
            base_seed=args.jovo_density_scale_base_seed,
        )
        return

    if args.trial == "jovo-density-scale-plot":
        do_jovo_density_scale_plot(
            output_png=args.jovo_density_scale_output_png,
            output_csv=args.jovo_density_scale_output_csv,
        )
        return

    nstreams = args.nstreams if args.nstreams is not None else args.ntrees
    do_jovo(
        data_dir=args.data_dir,
        train_split=args.train_split,
        ydf_use_slow_engine=not args.ydf_fast_engine,
        trial=args.trial,
        ntrees=args.ntrees,
        nstreams=nstreams,
        max_features=args.max_features,
    )


if __name__ == "__main__":
    main()
