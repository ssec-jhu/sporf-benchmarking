import ydf  # Yggdrasil Decision Forests
import pandas as pd  # Used for loading and manipulating small datasets
import numpy as np
import cupy as cp
import time
try:
    import cudf
except Exception:
    cudf = None

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from cuml.ensemble import RandomForestClassifier as curfc
from cuml.ensemble import SPORFClassifier as sporfc
from cuml.ensemble import SPORFRegressor as sporfr
from cuml.testing.utils import get_handle


def print_result(result):
    print(f"Hyperparameters: {result['hyperparameters']}")
    print(f"YDF RF training time: {result['ydfrf_train_time']:.2f} seconds")
    print(f"YDF SPORF training time: {result['ydfsporf_train_time']:.2f} seconds")
    print(f"cuML training time: {result['cuml_train_time']:.2f} seconds")
    print(f"SPORF training time: {result['sporf_train_time']:.2f} seconds")
    print()
    print(f"YDF RF prediction time: {result['ydfrf_predict_time']:.2f} seconds")
    print(f"YDF SPORF prediction time: {result['ydfsporf_predict_time']:.2f} seconds")
    print(f"cuML prediction time: {result['cuml_predict_time']:.2f} seconds")
    print(f"SPORF prediction time: {result['sporf_predict_time']:.2f} seconds")
    print()
    print(f"YDF RF Test accuracy: {result['ydfrf_accuracy']:.4f}")
    print(f"YDF SPORF Test accuracy: {result['ydfsporf_accuracy']:.4f}")
    print(f"cuML Test accuracy: {result['cuml_accuracy']:.4f}")
    print(f"SPORF Test accuracy: {result['sporf_accuracy']:.4f}")

    # Data preparation / validation timings
    print()
    print(f"Data prep total: {result['data_prep_time']:.2f} seconds")


def do_trial(label, n_train):
    # Generate synthetic classification data (off the clock, optional)
    data_args = {
        "n_samples": int(n_train / 0.8),  # generate extra samples to account for train/test split
        "n_features": 40,
        "n_clusters_per_class": 1,
        "n_informative": 20,
        "random_state": 123,
        "n_classes": 2,
    }
    X, y = make_classification(**data_args)

    # Split into train/test (50/50) preserving class proportions
    x_tr, x_ts, y_tr, y_ts = train_test_split(
        X, y, train_size=0.8, random_state=123, stratify=y
    )

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
    x_train = np.ascontiguousarray(x_tr.astype(np.float32))
    y_train = np.ascontiguousarray(y_tr.astype(np.int32))
    x_test = np.ascontiguousarray(x_ts.astype(np.float32))
    y_test = np.ascontiguousarray(y_ts.astype(np.int32))

    # Start YDF prep timer (include array contiguization that feeds YDF)
    t_vd0 = time.perf_counter()

    # Prepare YDF inputs and build training dict
    train_dict = {f"f{i}": x_train[:, i] for i in range(x_train.shape[1])}
    train_dict["foo"] = y_train
    # Some ydf versions accept a plain dict for training; avoid calling
    # dataset.create_vertical_dataset which may not exist in this ydf build.
    # We'll pass `train_dict` directly to the learner. Measure dict build time.
    t_vd = time.perf_counter() - t_vd0

    t_prep = time.perf_counter() - t_p0

    # ---------- TRAIN-ONLY (timed; reuse ydf_vd and X_dev across repeats) ----------
    # YDF train-only (VerticalDataset already built)
    num_features = x_train.shape[1]
    num_trees = 100
    num_projections = 5
    bootstrap_size_ratio = 0.8
    max_depth = 18
    min_samples_leaf = 2

    # ydf_learner = ydf.RandomForestLearner(label="label", num_trees=10)
    t0 = time.perf_counter()
    ydfrf_args = {
        "label": "foo",
        "bootstrap_size_ratio": bootstrap_size_ratio,
        "max_depth": max_depth,
        # max_num_nodes=-1,
        "min_examples": min_samples_leaf,
        "num_trees": num_trees,
        # discretize_numerical_columns=True
    }
    ydfrf_learner = ydf.RandomForestLearner(**ydfrf_args)
    ydfrf_model = ydfrf_learner.train(train_dict)
    t_ydfrf_train = time.perf_counter() - t0

    t0 = time.perf_counter()
    ydfsporf_args = {
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
        "sparse_oblique_projection_density_factor": 0.5 * num_features,
        "sparse_oblique_weights": "BINARY",
        # discretize_numerical_columns=True
    }
    ydfsporf_learner = ydf.RandomForestLearner(**ydfsporf_args)
    ydfsporf_model = ydfsporf_learner.train(train_dict)
    t_ydfsporf_train = time.perf_counter() - t0

    # prepare cuML handle/streams before constructing model
    n_streams = 1
    handle, streams = get_handle(True, n_streams=n_streams)

    # cuML train-only (device data already present)
    t0 = time.perf_counter()
    cuml_args = {
        #max_features=max_features,
        #max_samples=max_samples,
        "n_bins": 16,
        "split_criterion": 0,
        "min_samples_leaf": 2,
        #random_state=123,
        "n_streams": n_streams,
        "n_estimators": num_trees,
        "handle": handle,
        "max_leaves": -1,
        "max_depth": max_depth,
    }
    cuml_model = curfc(**cuml_args)
    cuml_model.fit(x_train, y_train)
    t_cuml_train = time.perf_counter() - t0

    t0 = time.perf_counter()
    sporf_args = {
        "max_features": num_projections / num_features,
        "max_samples": bootstrap_size_ratio,
        "n_bins": 16,
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
    sporf_model.fit(x_train, y_train)
    t_sporf_train = time.perf_counter() - t0


    # start timer for YDF prediction
    t0 = time.perf_counter()
    # Prepare test/eval dicts for YDF (CPU-side numpy arrays)
    test_ds = {f"f{i}": x_test[:, i] for i in range(x_test.shape[1])}
    eval_ds = {f"f{i}": x_test[:, i] for i in range(x_test.shape[1])}
    eval_ds["foo"] = y_test
    ydfrf_predictions = ydfrf_model.predict_class(test_ds).astype(int)
    t_ydfrf_predict = time.perf_counter() - t0

    ydfrf_evaluation = ydfrf_model.evaluate(eval_ds)

    t0 = time.perf_counter()
    # Prepare test/eval dicts for YDF (CPU-side numpy arrays)
    test_ds = {f"f{i}": x_test[:, i] for i in range(x_test.shape[1])}
    eval_ds = {f"f{i}": x_test[:, i] for i in range(x_test.shape[1])}
    eval_ds["foo"] = y_test
    ydfsporf_predictions = ydfsporf_model.predict_class(test_ds).astype(int)
    t_ydfsporf_predict = time.perf_counter() - t0

    ydfsporf_evaluation = ydfsporf_model.evaluate(eval_ds)

    t0 = time.perf_counter()
    cuml_preds = cuml_model.predict(x_test, predict_model="GPU")
    t_cuml_pred = time.perf_counter() - t0

    t0 = time.perf_counter()
    sp_preds = sporf_model.predict(x_test, predict_model="CPU")
    t_sporf_pred = time.perf_counter() - t0

    ydfrf_acc = accuracy_score(y_test, ydfrf_predictions)
    ydfsporf_acc = accuracy_score(y_test, ydfsporf_predictions)
    cuml_acc = accuracy_score(y_test, cuml_preds)
    sporf_acc = accuracy_score(y_test, sp_preds)

    # Query individual evaluation metrics
    hyperparameters = {"data_args": data_args | {"n_train": n_train}, "ydfrf_args": ydfrf_args, "ydfsporf_args": ydfsporf_args, "cuml_args": cuml_args, "sporf_args": sporf_args}
    result = {
        "label": label,
        "hyperparameters": hyperparameters,
        "ydfrf_train_time": t_ydfrf_train,
        "ydfsporf_train_time": t_ydfsporf_train,
        "cuml_train_time": t_cuml_train,
        "sporf_train_time": t_sporf_train,
        "ydfrf_predict_time": t_ydfrf_predict,
        "ydfsporf_predict_time": t_ydfsporf_predict,
        "cuml_predict_time": t_cuml_pred,
        "sporf_predict_time": t_sporf_pred,
        "ydfrf_accuracy": ydfrf_acc,
        "ydfsporf_accuracy": ydfsporf_acc,
        "cuml_accuracy": cuml_acc,
        "sporf_accuracy": sporf_acc,
        "data_prep_time": t_prep,
    }
    return result

results = []

for n_train in [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]:
    print(f"Running trial with n_train={n_train}...")
    result = do_trial(f"n_train={n_train}", n_train)
    print_result(result)
    results.append(result)

for res in results:
    print("-" * 40)
    print_result(res)
