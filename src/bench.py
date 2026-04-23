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

from cuml.ensemble import SPORFClassifier as sporfc
from cuml.ensemble import SPORFRegressor as sporfr
from cuml.testing.utils import get_handle


# Generate data
X, y = make_classification(
    n_samples=int(13_000_000 / 0.8),  # generate extra samples to account for train/test split
    n_features=40,
    n_clusters_per_class=1,
    n_informative=20,
    random_state=123,
    n_classes=2,
)

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

X_dev_train = x_train # cudf_train.drop(columns=["foo"]).astype(np.float32)  # cp.asarray(x_tr.astype(np.float32))
y_dev_train = y_train # cudf_train["foo"].astype(np.int32)  # cp.asarray(y_tr.astype(np.int32))

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
num_trees = 10
num_projections = 5
bootstrap_size_ratio = 0.8
max_depth = 18
min_samples_leaf = 2

# ydf_learner = ydf.RandomForestLearner(label="label", num_trees=10)
t0 = time.perf_counter()
ydf_learner = ydf.RandomForestLearner(
    label="foo",
    bootstrap_size_ratio=bootstrap_size_ratio,
    max_depth=max_depth,
    # max_num_nodes=-1,
    min_examples=min_samples_leaf,
    num_trees=num_trees,
    split_axis="SPARSE_OBLIQUE",
    sparse_oblique_max_num_projections=num_projections,
    sparse_oblique_num_projections_exponent=0.0,
    sparse_oblique_normalization="NONE",
    sparse_oblique_projection_density_factor=0.5 * num_features,
    sparse_oblique_weights="BINARY",
    # discretize_numerical_columns=True
)
ydf_model = ydf_learner.train(train_dict)
t_ydf_train = time.perf_counter() - t0

# prepare cuML handle/streams before constructing model
n_streams = 1
handle, streams = get_handle(True, n_streams=n_streams)

# cuML train-only (device data already present)
t0 = time.perf_counter()
sporf_model = sporfc(
    # max_features=num_projections,
    max_features=num_projections / num_features,
    max_samples=bootstrap_size_ratio,
    n_bins=16,
    split_criterion=0,
    min_samples_leaf=min_samples_leaf,
    # random_state=123,
    n_streams=n_streams,
    n_estimators=num_trees,
    handle=handle,
    max_leaves=-1,
    max_depth=max_depth,
    verbose=True
)
# measure construction time and training time separately
t_sporf_construct = time.perf_counter() - t0
t_fit_start = time.perf_counter()
sporf_model.fit(X_dev_train, y_dev_train)
t_sporf_train = time.perf_counter() - t_fit_start


# ds_path = "https://raw.githubusercontent.com/google/yggdrasil-decision-forests/main/yggdrasil_decision_forests/test_data/dataset"


# start timer for YDF prediction
t_start = time.perf_counter()
# Prepare test/eval dicts for YDF (CPU-side numpy arrays)
test_ds = {f"f{i}": x_test[:, i] for i in range(x_test.shape[1])}
eval_ds = {f"f{i}": x_test[:, i] for i in range(x_test.shape[1])}
eval_ds["foo"] = y_test
ydf_predictions = ydf_model.predict_class(test_ds).astype(int)
t_ydf_predict = time.perf_counter() - t_start

ydf_evaluation = ydf_model.evaluate(eval_ds)

t_start = time.perf_counter()
sp_preds = sporf_model.predict(x_test, predict_model="CPU")
t_sporf_pred = time.perf_counter() - t_start

ydf_acc = accuracy_score(y_test, ydf_predictions)
sporf_acc = accuracy_score(y_test, sp_preds)

# Query individual evaluation metrics
print(f"YDF training time: {t_ydf_train:.2f} seconds")
print(f"SPORF training time: {t_sporf_train:.2f} seconds")
print(f"SPORF construction time: {t_sporf_construct:.2f} seconds")
print()
print(f"YDF prediction time: {t_ydf_predict:.2f} seconds")
print(f"SPORF prediction time: {t_sporf_pred:.2f} seconds")
print()
print(f"ydf Test accuracy: {ydf_acc:.4f}")
print(f"SPORF Test accuracy: {sporf_acc:.4f}")

# Data preparation / validation timings
print()
print(f"Data prep total: {t_prep:.2f} seconds (VerticalDataset: {t_vd:.2f}, H2D: {t_h2d:.2f})")

# Show the full evaluation report
# print("ydf Full evaluation report:")
# print(ydf_evaluation)
