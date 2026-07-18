import argparse
from pathlib import Path
import ydf  # Yggdrasil Decision Forests
import pandas as pd  # Used for loading and manipulating small datasets
import numpy as np
import cupy as cp
import time
import pickle

try:
    import cudf
except Exception:
    cudf = None

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from cuml.ensemble import SPORFClassifier as sporfc
from cuml.ensemble import SPORFRegressor as sporfr
from cuml.testing.utils import get_handle


PROJECT_ROOT = Path(__file__).resolve().parent.parent
HIGGS_PATH = PROJECT_ROOT / "data" / "higgs" / "HIGGS.csv"


def print_result(result):
    print(f"Hyperparameters: {result['hyperparameters']}")
    print(f"YDF training time: {result['ydf_train_time']:.2f} seconds")
    print(f"SPORF training time: {result['sporf_train_time']:.2f} seconds")
    print()
    print(f"YDF prediction time: {result['ydf_predict_time']:.2f} seconds")
    print(f"SPORF prediction time: {result['sporf_predict_time']:.2f} seconds")
    print()
    print(f"ydf Test accuracy: {result['ydf_accuracy']:.4f}")
    print(f"SPORF Test accuracy: {result['sporf_accuracy']:.4f}")
    print(f"SPORF Unpickled Test accuracy: {result['sporf_unpickled_accuracy']:.4f}")

    # Data preparation / validation timings
    print()
    print(f"Data prep total: {result['data_prep_time']:.2f} seconds")


def print_quick_result(result):
    print(f"Hyperparameters: {result['hyperparameters']}")
    print(f"SPORF training time: {result['sporf_train_time']:.2f} seconds")
    print()
    print(f"SPORF prediction time: {result['sporf_predict_time']:.2f} seconds")
    print()
    print(f"SPORF Test accuracy: {result['sporf_accuracy']:.4f}")
    print(f"SPORF Unpickled Test accuracy: {result['sporf_unpickled_accuracy']:.4f}")
    print()
    print(f"Data prep total: {result['data_prep_time']:.2f} seconds")


def do_trial(x_tr, y_tr, x_ts, y_ts, data_args, n_train, ydf_use_slow_engine=False):
    # # Split into train/test preserving class proportions
    # x_tr, x_ts, y_tr, y_ts = train_test_split(
    #     X, y, train_size=train_split, random_state=123, stratify=y
    # )

    # Create cudf fast-path inputs from training split (off the clock, optional)
    if cudf is not None:
        cudf_train = cudf.DataFrame({f"f{i}": x_tr[:, i] for i in range(x_tr.shape[1])})
        cudf_train["foo"] = cudf.Series(y_tr)
    else:
        cudf_train = None

    # ---------- PREPROCESS (do once; time separately) ----------
    t_p0 = time.perf_counter()

    # Validate shapes
    for arr, name, dims in ((x_tr, "x_tr", 2), (x_ts, "x_ts", 2)):
        if arr.ndim != dims: raise ValueError(f"{name} must be {dims}-D")
    for arr, name, dims in ((y_tr, "y_tr", 1), (y_ts, "y_ts", 1)):
        if arr.ndim != dims:
            raise ValueError(f"{name} must be {dims}-D")

    # Convert types and ensure contiguous layout for performance
    x_train = np.ascontiguousarray(x_tr.astype(np.float32, copy=False))
    y_train = np.ascontiguousarray(y_tr.astype(np.int32, copy=False))
    x_test = np.ascontiguousarray(x_ts.astype(np.float32, copy=False))
    y_test = np.ascontiguousarray(y_ts.astype(np.int32, copy=False))

    # X_dev_train = x_train
    # y_dev_train = y_train

    # Start YDF prep timer (include array contiguization that feeds YDF)
    t_vd0 = time.perf_counter()

    # Prepare YDF inputs and build training dict
    train_dict = {f"f{i}": x_train[:, i] for i in range(x_train.shape[1])}
    train_dict["foo"] = y_train
    # Some ydf versions accept a plain dict for training; avoid calling
    # dataset.create_vertical_dataset which may not exist in this ydf build.
    # We'll pass `train_dict` directly to the learner. Measure dict build time.
    t_vd = time.perf_counter() - t_vd0

    # # Quick class-count on training labels (timed)
    # t_unique0 = time.perf_counter()
    # unique_vals, unique_counts = np.unique(y_train, return_counts=True)
    # t_unique = time.perf_counter() - t_unique0

    # Prepare cuML device arrays (training data)
    t_h2d0 = time.perf_counter()
    t_h2d = time.perf_counter() - t_h2d0

    t_prep = time.perf_counter() - t_p0

    # ---------- TRAIN-ONLY (timed; reuse ydf_vd and X_dev across repeats) ----------
    # YDF train-only (VerticalDataset already built)
    num_features = x_train.shape[1]
    num_trees = 4
    num_projections = 5
    projection_density = 0.5
    bootstrap_size_ratio = 0.8
    max_depth = 18
    min_samples_leaf = 2
    n_bins = 128

    # ydf_learner = ydf.RandomForestLearner(label="label", num_trees=10)
    t0 = time.perf_counter()
    ydf_args = {
        "label": "foo",
        "bootstrap_size_ratio": bootstrap_size_ratio,
        "max_depth": max_depth,
        # max_num_nodes=-1,
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
    }
    ydf_learner = ydf.RandomForestLearner(**ydf_args)
    ydf_model = ydf_learner.train(train_dict)
    t_ydf_train = time.perf_counter() - t0

    # prepare cuML handle/streams before constructing model
    n_streams = 100
    handle, streams = get_handle(True, n_streams=n_streams)

    # cuML train-only (device data already present)
    t0 = time.perf_counter()
    sporf_args = {
        "max_features": num_projections / num_features,
        "max_samples": bootstrap_size_ratio,
        "density": projection_density,
        "n_bins": n_bins,
        "split_criterion": 0,
        "min_samples_leaf": min_samples_leaf,
        "n_streams": n_streams,
        "n_estimators": num_trees,
        "handle": handle,
        "max_leaves": -1,
        "max_depth": max_depth,
        "verbose": False
    }
    sporf_model = sporfc(**sporf_args)
    # measure construction time and training time separately
    t_sporf_construct = time.perf_counter() - t0
    t_fit_start = time.perf_counter()
    sporf_model.fit(x_train, y_train)
    t_sporf_train = time.perf_counter() - t_fit_start


    # ds_path = "https://raw.githubusercontent.com/google/yggdrasil-decision-forests/main/yggdrasil_decision_forests/test_data/dataset"


    # start timer for YDF prediction
    t_start = time.perf_counter()
    # Prepare test/eval dicts for YDF (CPU-side numpy arrays)
    test_ds = {f"f{i}": x_test[:, i] for i in range(x_test.shape[1])}
    eval_ds = {f"f{i}": x_test[:, i] for i in range(x_test.shape[1])}
    eval_ds["foo"] = y_test
    ydf_predictions = ydf_model.predict_class(
        test_ds, use_slow_engine=ydf_use_slow_engine
    ).astype(int)
    t_ydf_predict = time.perf_counter() - t_start

    ydf_evaluation = ydf_model.evaluate(
        eval_ds, use_slow_engine=ydf_use_slow_engine
    )

    t_start = time.perf_counter()
    sp_preds = sporf_model.predict(x_test, predict_model="CPU")
    t_sporf_pred = time.perf_counter() - t_start

    sporf_model_unpickled = pickle.loads(pickle.dumps(sporf_model)) # test that model is pickleable without error
    spu_preds = sporf_model_unpickled.predict(x_test, predict_model="CPU")

    ydf_acc = accuracy_score(y_test, ydf_predictions)
    sporf_acc = accuracy_score(y_test, sp_preds)
    sporfu_acc = accuracy_score(y_test, spu_preds)

    # Query individual evaluation metrics
    hyperparameters = {
        "data_args": data_args | {"n_train": n_train},
        "ydf_args": ydf_args | {"use_slow_engine": ydf_use_slow_engine},
        "sporf_args": sporf_args,
    }
    result = {
        "hyperparameters": hyperparameters,
        "ydf_train_time": t_ydf_train,
        "sporf_train_time": t_sporf_train,
        "ydf_predict_time": t_ydf_predict,
        "sporf_predict_time": t_sporf_pred,
        "ydf_accuracy": ydf_acc,
        "sporf_accuracy": sporf_acc,
        "sporf_unpickled_accuracy": sporfu_acc,
        "data_prep_time": t_prep,
    }
    return result


def do_quick_trial(x_tr, y_tr, x_ts, y_ts, data_args, n_train):
    t_p0 = time.perf_counter()

    for arr, name, dims in ((x_tr, "x_tr", 2), (x_ts, "x_ts", 2)):
        if arr.ndim != dims:
            raise ValueError(f"{name} must be {dims}-D")
    for arr, name, dims in ((y_tr, "y_tr", 1), (y_ts, "y_ts", 1)):
        if arr.ndim != dims:
            raise ValueError(f"{name} must be {dims}-D")

    x_train = np.ascontiguousarray(x_tr.astype(np.float32, copy=False))
    y_train = np.ascontiguousarray(y_tr.astype(np.int32, copy=False))
    x_test = np.ascontiguousarray(x_ts.astype(np.float32, copy=False))
    y_test = np.ascontiguousarray(y_ts.astype(np.int32, copy=False))

    t_prep = time.perf_counter() - t_p0

    num_features = x_train.shape[1]
    num_trees = 18
    num_projections = 5
    projection_density = 0.5
    bootstrap_size_ratio = 0.8
    max_depth = 18
    min_samples_leaf = 2
    n_bins = 128

    n_streams = 100
    handle, streams = get_handle(True, n_streams=n_streams)

    t0 = time.perf_counter()
    sporf_args = {
        "max_features": num_projections / num_features,
        "max_samples": bootstrap_size_ratio,
        "density": projection_density,
        "n_bins": n_bins,
        "split_criterion": 0,
        "min_samples_leaf": min_samples_leaf,
        "n_streams": n_streams,
        "n_estimators": num_trees,
        "handle": handle,
        "max_leaves": -1,
        "max_depth": max_depth,
        "verbose": False
    }
    sporf_model = sporfc(**sporf_args)
    t_fit_start = time.perf_counter()
    sporf_model.fit(x_train, y_train)
    t_sporf_train = time.perf_counter() - t_fit_start

    t_start = time.perf_counter()
    sp_preds = sporf_model.predict(x_test, predict_model="CPU")
    t_sporf_pred = time.perf_counter() - t_start

    sporf_model_unpickled = pickle.loads(pickle.dumps(sporf_model))
    spu_preds = sporf_model_unpickled.predict(x_test, predict_model="CPU")

    sporf_acc = accuracy_score(y_test, sp_preds)
    sporfu_acc = accuracy_score(y_test, spu_preds)

    hyperparameters = {"data_args": data_args | {"n_train": n_train}, "sporf_args": sporf_args}
    result = {
        "hyperparameters": hyperparameters,
        "sporf_train_time": t_sporf_train,
        "sporf_predict_time": t_sporf_pred,
        "sporf_accuracy": sporf_acc,
        "sporf_unpickled_accuracy": sporfu_acc,
        "data_prep_time": t_prep,
    }
    return result


def do_classification():
    results = []

    for n_train in [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000, 15_000_000]:
        train_split = 0.8
        # Generate synthetic classification data
        data_args = {
            "n_samples": int(n_train / train_split),  # generate extra samples to account for train/test split
            "n_features": 40,
            "n_clusters_per_class": 1,
            "n_informative": 20,
            "random_state": 123,
            "n_classes": 2,
        }
        X, y = make_classification(**data_args)

        # Split into train/test preserving class proportions
        x_tr, x_ts, y_tr, y_ts = train_test_split(
            X, y, train_size=train_split, random_state=123, stratify=y
        )

        print(f"Running trial with n_train={n_train}...")
        result = do_trial(x_tr, y_tr, x_ts, y_ts, data_args, n_train)
        print_result(result)
        results.append(result)
    
    return results

def do_higgs():
    train_split = 0.8
    # Load Higgs dataset
    # ds_path = "https://raw.githubusercontent.com/google/yggdrasil-decision-forests/main/yggdrasil_decision_forests/test_data/dataset"
    ds_path = HIGGS_PATH
    df = pd.read_csv(ds_path, header=None)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values

    x_tr, x_ts, y_tr, y_ts = train_test_split(
        X, y, train_size=train_split, random_state=123, stratify=y
    )
    data_args = {"dataset": "Higgs"}
    result = do_trial(x_tr, y_tr, x_ts, y_ts, data_args, n_train=x_tr.shape[0])
    print_result(result)
    return result


def do_quick():
    train_split = 0.8
    ds_path = HIGGS_PATH
    df = pd.read_csv(ds_path, header=None)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values

    x_tr, x_ts, y_tr, y_ts = train_test_split(
        X, y, train_size=train_split, random_state=123, stratify=y
    )
    data_args = {"dataset": "Higgs"}
    result = do_quick_trial(x_tr, y_tr, x_ts, y_ts, data_args, n_train=x_tr.shape[0])
    print_quick_result(result)
    return result


def make_superwide_classification(
    n_samples,
    n_features,
    n_informative,
    random_state,
    signal_shift=0.15,
):
    rng = np.random.default_rng(random_state)
    y = np.arange(n_samples, dtype=np.int32) % 2
    rng.shuffle(y)

    X = rng.standard_normal((n_samples, n_features), dtype=np.float32)
    class_sign = (2 * y - 1).astype(np.float32)
    X[:, :n_informative] += signal_shift * class_sign[:, None]
    return X, y


def do_quick_superwide():
    train_split = 0.8
    n_train = 100
    n_features = 500_000

    data_args = {
        "n_samples": int(n_train / train_split),
        "n_features": n_features,
        "n_informative": n_features // 2,
        "random_state": 123,
        "n_classes": 2,
        "signal_shift": 0.15,
    }
    X, y = make_superwide_classification(
        n_samples=data_args["n_samples"],
        n_features=data_args["n_features"],
        n_informative=data_args["n_informative"],
        random_state=data_args["random_state"],
        signal_shift=data_args["signal_shift"],
    )

    x_tr, x_ts, y_tr, y_ts = train_test_split(
        X, y, train_size=train_split, random_state=123, stratify=y
    )
    result = do_quick_trial(x_tr, y_tr, x_ts, y_ts, data_args, n_train=x_tr.shape[0])
    print_quick_result(result)
    return result


def do_superwide():
    train_split = 0.8
    n_train = 100
    n_features = 500_000

    data_args = {
        "n_samples": int(n_train / train_split),
        "n_features": n_features,
        "n_informative": n_features // 2,
        "random_state": 123,
        "n_classes": 2,
        "signal_shift": 0.15,
    }
    X, y = make_superwide_classification(
        n_samples=data_args["n_samples"],
        n_features=data_args["n_features"],
        n_informative=data_args["n_informative"],
        random_state=data_args["random_state"],
        signal_shift=data_args["signal_shift"],
    )

    x_tr, x_ts, y_tr, y_ts = train_test_split(
        X, y, train_size=train_split, random_state=123, stratify=y
    )
    result = do_trial(
        x_tr,
        y_tr,
        x_ts,
        y_ts,
        data_args,
        n_train=x_tr.shape[0],
        ydf_use_slow_engine=True,
    )
    print_result(result)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=["classification", "higgs", "quick", "quick-superwide", "superwide"],
        required=True,
        help="Benchmark task to run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.task == "classification":
        do_classification()
    elif args.task == "higgs":
        do_higgs()
    elif args.task == "quick":
        do_quick()
    elif args.task == "quick-superwide":
        do_quick_superwide()
    elif args.task == "superwide":
        do_superwide()


if __name__ == "__main__":
    main()
