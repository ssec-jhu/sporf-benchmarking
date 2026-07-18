import argparse
import os
import math
import re
import subprocess
import sys
import tempfile

import numpy as np

from cuml.ensemble import SPORFClassifier
from cuml.testing.utils import get_handle


DEFAULT_MAX_FEATURES_CASES = [
    "sqrt",
    "log2",
    "None",
    "1",
    "2",
    "0.2",
    "0.5",
    "1.0",
    "bad",
]
DEFAULT_DENSITY_CASES = [
    "2",
    "2.0",
    "128",
    "128.0",
    "0.000290654107987",
    "0.0186018629112",
]


def make_data(n_samples, n_features, random_state):
    rng = np.random.default_rng(random_state)
    X = rng.standard_normal((n_samples, n_features), dtype=np.float32)
    y = np.arange(n_samples, dtype=np.int32) % 2
    rng.shuffle(y)

    signal_width = min(8, n_features)
    class_sign = (2 * y - 1).astype(np.float32)
    X[:, :signal_width] += 0.5 * class_sign[:, None]
    return np.ascontiguousarray(X), np.ascontiguousarray(y)


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


def parse_int_or_float(value):
    try:
        return int(value)
    except ValueError:
        return float(value)


def expected_projection_count(max_features, n_features):
    if isinstance(max_features, int):
        return max_features
    if isinstance(max_features, float):
        return int(max_features * n_features)
    if max_features == "sqrt":
        return int(math.sqrt(n_features))
    if max_features == "log2":
        return int(math.log2(n_features))
    if max_features is None:
        return n_features
    return "library-defined / expected rejection"


def run_with_captured_native_output(fn):
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
            fn()
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout, stdout_fd)
            os.dup2(saved_stderr, stderr_fd)
            os.close(saved_stdout)
            os.close(saved_stderr)
            captured.seek(0)
        return captured.read().decode(errors="replace")


def print_filtered_native_output(text, pattern):
    line_re = re.compile(pattern)
    for line in text.splitlines():
        if line_re.search(line):
            print(line)
            print_diagnostic_summary(line)


def parse_diagnostic_line(line):
    prefix = "SPORF hyperparameter diagnostics:"
    if not line.startswith(prefix):
        return None

    fields = {}
    for item in line[len(prefix) :].split(","):
        if "=" not in item:
            continue
        key, value = item.strip().split("=", 1)
        fields[key] = value
    return fields


def diagnostic_value(fields, key):
    value = fields.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return value


def format_number(value):
    if value is None:
        return "unavailable"
    if isinstance(value, str):
        return value
    return f"{value:g}"


def projection_attempt_variance(fields):
    observed_variance = diagnostic_value(fields, "projection_attempts_variance")
    if observed_variance is not None:
        return observed_variance

    min_attempts = diagnostic_value(fields, "projection_attempts_min")
    max_attempts = diagnostic_value(fields, "projection_attempts_max")
    if min_attempts is not None and min_attempts == max_attempts:
        return 0.0
    return None


def numeric_delta(empirical, intended):
    if isinstance(empirical, float) and isinstance(intended, float):
        return empirical - intended
    return None


def print_diagnostic_summary(line):
    fields = parse_diagnostic_line(line)
    if fields is None:
        return

    intended_nnz = diagnostic_value(fields, "expected_nnz")
    empirical_nnz = diagnostic_value(fields, "nnz_mean")
    intended_projections = diagnostic_value(fields, "projections_per_node_specified")
    empirical_projections = diagnostic_value(fields, "projection_attempts_mean")
    projection_variance = projection_attempt_variance(fields)

    print("TL;DR:")
    print(
        "  E[NNZ]: "
        f"intended={format_number(intended_nnz)}, "
        f"empirical_mean={format_number(empirical_nnz)}, "
        f"delta={format_number(numeric_delta(empirical_nnz, intended_nnz))}"
    )
    print(
        "  projections_per_node: "
        f"intended={format_number(intended_projections)}, "
        f"empirical_mean={format_number(empirical_projections)}, "
        f"delta={format_number(numeric_delta(empirical_projections, intended_projections))}, "
        f"empirical_variance={format_number(projection_variance)}"
    )


def filter_full_output(text, pattern):
    line_re = re.compile(pattern)
    metadata_prefixes = (
        "=",
        "CASE:",
        "n_features=",
        "density_specified=",
        "expected_nnz_if_fraction=",
        "max_features_specified=",
        "expected_projection_count_from_python_semantics=",
        "CASE FAILED:",
    )
    for line in text.splitlines():
        is_diagnostic = line_re.search(line)
        if line.startswith(metadata_prefixes) or is_diagnostic:
            print(line)
            if is_diagnostic:
                print_diagnostic_summary(line)


def argv_without_filter_flags(argv):
    filtered = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == "--filter-cuml-output":
            continue
        if arg == "--cuml-output-pattern":
            skip_next = True
            continue
        if arg.startswith("--cuml-output-pattern="):
            continue
        filtered.append(arg)
    return filtered


def run_filtered_subprocess(args):
    cmd = [sys.executable]
    if sys.flags.no_user_site:
        cmd.append("-s")
    cmd.extend(argv_without_filter_flags(sys.argv))
    env = dict(os.environ)
    env["SPORF_DIAGNOSTIC_FILTER_CHILD"] = "1"
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        check=False,
    )
    filter_full_output(result.stdout, args.cuml_output_pattern)
    return result.returncode


def fit_model(model, X, y, args):
    if not args.filter_cuml_output:
        model.fit(X, y)
        return

    text = run_with_captured_native_output(lambda: model.fit(X, y))
    print_filtered_native_output(text, args.cuml_output_pattern)


def run_case(name, X, y, args, max_features, density):
    n_streams = args.nstreams
    handle, streams = get_handle(True, n_streams=n_streams)
    params = {
        "max_features": max_features,
        "density": density,
        "n_estimators": args.ntrees,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "max_samples": args.max_samples,
        "n_bins": args.n_bins,
        "split_criterion": 0,
        "max_leaves": -1,
        "n_streams": n_streams,
        "handle": handle,
        "verbose": args.verbose,
    }

    print("=" * 88)
    print(f"CASE: {name}")
    print(f"n_features={X.shape[1]}")
    print(f"density_specified={density}")
    print(f"expected_nnz_if_fraction={density * X.shape[1]}")
    print(f"max_features_specified={max_features!r}")
    print(
        "expected_projection_count_from_python_semantics="
        f"{expected_projection_count(max_features, X.shape[1])}"
    )
    print(f"estimator_params={params}")
    print("-" * 88)

    try:
        model = SPORFClassifier(**params)
        fit_model(model, X, y, args)
    except Exception as exc:
        print(f"CASE FAILED: {type(exc).__name__}: {exc}")
    print()


def density_trials(X, y, args):
    max_features = args.projections / X.shape[1]
    for density in args.densities:
        run_case(
            name=f"density={density}",
            X=X,
            y=y,
            args=args,
            max_features=max_features,
            density=density,
        )


def projection_trials(X, y, args):
    for projections in args.projection_counts:
        run_case(
            name=f"projections={projections}",
            X=X,
            y=y,
            args=args,
            max_features=projections / X.shape[1],
            density=args.density,
        )


def polymorphism_trials(X, y, args):
    for max_features in args.max_features_cases:
        run_case(
            name=f"max_features={max_features!r}",
            X=X,
            y=y,
            args=args,
            max_features=max_features,
            density=args.density,
        )


def density_polymorphism_trials(X, y, args):
    max_features = args.projections / X.shape[1]
    for density in args.density_cases:
        run_case(
            name=f"density={density!r}",
            X=X,
            y=y,
            args=args,
            max_features=max_features,
            density=density,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trial",
        choices=[
            "all",
            "density",
            "projections",
            "polymorphism",
            "density-polymorphism",
        ],
        default="all",
        help="Diagnostic trial group to run.",
    )
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--n-features", type=int, default=32)
    parser.add_argument("--random-state", type=int, default=123)
    parser.add_argument("--ntrees", type=int, default=1)
    parser.add_argument("--nstreams", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--max-samples", type=float, default=0.8)
    parser.add_argument("--n-bins", type=int, default=32)
    parser.add_argument(
        "--verbose",
        default=True,
        help=(
            "cuML verbose value. Default True maps to debug in this cuML build. "
            "Use 5 for integer debug verbosity."
        ),
    )
    parser.add_argument(
        "--filter-cuml-output",
        dest="filter_cuml_output",
        action="store_true",
        default=True,
        help=(
            "Run the diagnostic in a subprocess, capture native cuML stdout/stderr, "
            "and print only case metadata plus lines matching --cuml-output-pattern. "
            "This is the default."
        ),
    )
    parser.add_argument(
        "--no-filter-cuml-output",
        dest="filter_cuml_output",
        action="store_false",
        help="Print raw cuML output without subprocess filtering.",
    )
    parser.add_argument(
        "--cuml-output-pattern",
        default=r"^SPORF hyperparameter diagnostics:",
        help="Regex used when --filter-cuml-output is enabled.",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=0.5,
        help="Default density for non-density sweeps.",
    )
    parser.add_argument(
        "--projections",
        type=int,
        default=5,
        help="Default projection count for non-projection sweeps.",
    )
    parser.add_argument(
        "--densities",
        type=float,
        nargs="+",
        default=[1 / 32, 2 / 32, 4 / 32, 0.5, 1.0],
        help="Density values to sweep.",
    )
    parser.add_argument(
        "--projection-counts",
        type=int,
        nargs="+",
        default=[1, 2, 5, 10],
        help="Projection counts to sweep.",
    )
    parser.add_argument(
        "--max-features-cases",
        type=parse_max_features,
        nargs="+",
        default=[parse_max_features(value) for value in DEFAULT_MAX_FEATURES_CASES],
        help=(
            "max_features argument polymorphism cases. Values that parse as "
            "int/float become int/float; 'None' becomes None; others stay strings."
        ),
    )
    parser.add_argument(
        "--density-cases",
        type=parse_int_or_float,
        nargs="+",
        default=[parse_int_or_float(value) for value in DEFAULT_DENSITY_CASES],
        help=(
            "density argument polymorphism cases. Values without decimal points "
            "become ints; values with decimal points become floats."
        ),
    )

    args = parser.parse_args()
    if isinstance(args.verbose, str):
        if args.verbose.lower() == "true":
            args.verbose = True
        elif args.verbose.lower() == "false":
            args.verbose = False
        else:
            try:
                args.verbose = int(args.verbose)
            except ValueError:
                parser.error("--verbose must be true, false, or an integer")
    if args.n_samples < 2:
        parser.error("--n-samples must be at least 2")
    if args.n_features < 1:
        parser.error("--n-features must be at least 1")
    if args.ntrees < 1:
        parser.error("--ntrees must be at least 1")
    if args.nstreams < 1:
        parser.error("--nstreams must be at least 1")
    args.max_features_cases = [
        value if not isinstance(value, str) else parse_max_features(value)
        for value in args.max_features_cases
    ]
    return args


def main():
    args = parse_args()
    if (
        args.filter_cuml_output
        and os.environ.get("SPORF_DIAGNOSTIC_FILTER_CHILD") != "1"
    ):
        raise SystemExit(run_filtered_subprocess(args))

    X, y = make_data(args.n_samples, args.n_features, args.random_state)

    if args.trial in {"all", "density"}:
        density_trials(X, y, args)
    if args.trial in {"all", "projections"}:
        projection_trials(X, y, args)
    if args.trial in {"all", "polymorphism"}:
        polymorphism_trials(X, y, args)
    if args.trial in {"all", "density-polymorphism"}:
        density_polymorphism_trials(X, y, args)


if __name__ == "__main__":
    main()
