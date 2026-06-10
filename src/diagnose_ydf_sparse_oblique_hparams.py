import argparse
import math
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import ydf


FEATURE_RE = re.compile(r"\bf(\d+)\b")
OBLIQUE_HINT_RE = re.compile(r"oblique|projection|sparse", re.IGNORECASE)
LINEAR_EXPR_RE = re.compile(r"[+*]")
ATTRIBUTES_RE = re.compile(r"attributes=\[([^\]]*)\]")
WEIGHTS_RE = re.compile(r"weights=\[([^\]]*)\]")
SCALAR_TYPES = (str, bytes, int, float, bool, type(None))


def run_with_suppressed_native_output(fn, suppress):
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


def make_data(n_samples, n_features, signal_nnz, random_state):
    rng = np.random.default_rng(random_state)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)

    signal_nnz = min(signal_nnz, n_features)
    weights = rng.choice([-1.0, 1.0], size=signal_nnz).astype(np.float32)
    score = X[:, :signal_nnz] @ weights
    score += 0.1 * rng.standard_normal(n_samples).astype(np.float32)
    y = (score > np.median(score)).astype(np.int32)

    return np.ascontiguousarray(X), np.ascontiguousarray(y)


def make_ydf_dict(X, y=None):
    data = {f"f{i}": X[:, i] for i in range(X.shape[1])}
    if y is not None:
        data["label"] = y
    return data


def base_learner_args(args, intended_nnz):
    return {
        "label": "label",
        "num_trees": 1,
        "split_axis": "SPARSE_OBLIQUE",
        "sparse_oblique_max_num_projections": args.num_projections,
        "sparse_oblique_num_projections_exponent": args.num_projections_exponent,
        "sparse_oblique_normalization": "NONE",
        "sparse_oblique_projection_density_factor": intended_nnz,
        "sparse_oblique_weights": "BINARY",
        "bootstrap_size_ratio": args.bootstrap_size_ratio,
        "max_depth": args.max_depth,
        "min_examples": args.min_examples,
    }


def train_model(train_ds, learner_args, suppress_output=True):
    try:
        learner = ydf.RandomForestLearner(**learner_args)
        return run_with_suppressed_native_output(
            lambda: learner.train(train_ds),
            suppress_output,
        )
    except TypeError as exc:
        if "random_seed" not in str(exc):
            raise
        learner_args = dict(learner_args)
        learner_args.pop("random_seed", None)
        learner = ydf.RandomForestLearner(**learner_args)
        return run_with_suppressed_native_output(
            lambda: learner.train(train_ds),
            suppress_output,
        )


def model_description(model):
    describe = getattr(model, "describe", None)
    if describe is None:
        return ""
    value = describe()
    return "" if value is None else str(value)


def read_serialized_text(model):
    texts = []
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model"
        save = getattr(model, "save", None)
        if save is None:
            return texts
        save(str(path))
        for file_path in path.rglob("*"):
            if not file_path.is_file():
                continue
            try:
                raw = file_path.read_bytes()
            except OSError:
                continue
            if b"\x00" in raw[:4096]:
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue
            texts.append((str(file_path.relative_to(path)), text))
    return texts


def save_model_for_inspection(model, output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing model dir: {output_dir}")
    save = getattr(model, "save", None)
    if save is None:
        raise AttributeError("YDF model does not expose save(...)")
    save(str(output_dir))
    return output_dir


def serialized_file_manifest(model_dir, max_preview_bytes):
    rows = []
    for file_path in sorted(Path(model_dir).rglob("*")):
        if not file_path.is_file():
            continue
        raw = file_path.read_bytes()
        has_nul = b"\x00" in raw[:4096]
        preview = ""
        if not has_nul:
            try:
                preview = raw[:max_preview_bytes].decode("utf-8", errors="replace")
            except UnicodeDecodeError:
                preview = ""
        rows.append(
            {
                "path": str(file_path.relative_to(model_dir)),
                "bytes": len(raw),
                "looks_binary": has_nul,
                "preview": preview,
            }
        )
    return rows


def print_serialized_manifest(model_dir, max_preview_bytes):
    print(f"SERIALIZED_MODEL_DIR={model_dir}")
    for row in serialized_file_manifest(model_dir, max_preview_bytes):
        print(
            "SERIALIZED_FILE: "
            f"path={row['path']} bytes={row['bytes']} "
            f"looks_binary={row['looks_binary']}"
        )
        if row["preview"]:
            print("SERIALIZED_PREVIEW_BEGIN")
            print(row["preview"])
            print("SERIALIZED_PREVIEW_END")


def candidate_projection_lines(text):
    lines = []
    for line in text.splitlines():
        features = FEATURE_RE.findall(line)
        if len(set(features)) < 2:
            continue
        if not (OBLIQUE_HINT_RE.search(line) or LINEAR_EXPR_RE.search(line)):
            continue
        lines.append(line.strip())
    return lines


def object_fields(obj):
    if isinstance(obj, SCALAR_TYPES):
        return []
    fields = []
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if callable(value):
            continue
        fields.append((name, value))
    return fields


def field_names(obj):
    return [name for name, _ in object_fields(obj)]


def sized_sequence(value):
    if isinstance(value, SCALAR_TYPES):
        return None
    if isinstance(value, dict):
        return None
    if not hasattr(value, "__len__") or not hasattr(value, "__iter__"):
        return None
    try:
        return list(value)
    except TypeError:
        return None


def count_repr_list_items(obj, field_name):
    pattern = ATTRIBUTES_RE if field_name == "attributes" else WEIGHTS_RE
    match = pattern.search(repr(obj))
    if match is None:
        return None
    body = match.group(1).strip()
    if not body:
        return 0
    return len([item for item in body.split(",") if item.strip()])


def parse_repr_int_list(obj, field_name):
    pattern = ATTRIBUTES_RE if field_name == "attributes" else WEIGHTS_RE
    match = pattern.search(repr(obj))
    if match is None:
        return None
    body = match.group(1).strip()
    if not body:
        return []
    values = []
    for item in body.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


def extract_oblique_condition_attributes(condition):
    if condition is None:
        return None

    cls_name = type(condition).__name__.lower()
    field_map = {name: value for name, value in object_fields(condition)}
    if "oblique" not in cls_name and "attributes" not in field_map:
        return None

    values = sized_sequence(field_map.get("attributes"))
    if values is not None:
        return [int(value) for value in values]

    return parse_repr_int_list(condition, "attributes")


def extract_oblique_condition_nnz(condition):
    if condition is None:
        return None

    cls_name = type(condition).__name__.lower()
    field_map = {name: value for name, value in object_fields(condition)}
    if "oblique" not in cls_name and not {
        "attributes",
        "weights",
        "features",
        "feature_idxs",
        "attribute_idxs",
    }.intersection(field_map):
        return None

    for name in (
        "attributes",
        "features",
        "feature_idxs",
        "attribute_idxs",
        "attribute_indices",
        "feature_indices",
    ):
        values = sized_sequence(field_map.get(name))
        if values is not None:
            return len(values), f"tree_api:{type(condition).__name__}.{name}"

    weights = sized_sequence(field_map.get("weights"))
    if weights is not None:
        return len(weights), f"tree_api:{type(condition).__name__}.weights"

    attribute_count = count_repr_list_items(condition, "attributes")
    if attribute_count is not None:
        return attribute_count, f"tree_api:{type(condition).__name__}.repr.attributes"

    weight_count = count_repr_list_items(condition, "weights")
    if weight_count is not None:
        return weight_count, f"tree_api:{type(condition).__name__}.repr.weights"

    return None


def root_condition(model):
    get_tree = getattr(model, "get_tree", None)
    if get_tree is None:
        return None
    tree = get_tree(0)
    root = getattr(tree, "root", None)
    if root is None or getattr(root, "is_leaf", False):
        return None
    return getattr(root, "condition", None)


def child_objects(obj):
    children = []
    preferred_names = (
        "root",
        "pos_child",
        "neg_child",
        "positive_child",
        "negative_child",
        "children",
    )
    field_map = {name: value for name, value in object_fields(obj)}
    for name in preferred_names:
        if name not in field_map:
            continue
        value = field_map[name]
        seq = sized_sequence(value)
        if seq is None:
            children.append(value)
        else:
            children.extend(seq)
    return children


def extract_tree_api_nnz(model):
    get_tree = getattr(model, "get_tree", None)
    if get_tree is None:
        return [], []

    try:
        tree = get_tree(0)
    except Exception as exc:
        return [], [("tree_api", f"get_tree(0) failed: {type(exc).__name__}: {exc}", 0)]

    observations = []
    matching = []
    stack = [tree]
    seen = set()
    while stack:
        obj = stack.pop()
        obj_id = id(obj)
        if obj_id in seen or isinstance(obj, SCALAR_TYPES):
            continue
        seen.add(obj_id)

        condition = getattr(obj, "condition", obj)
        extracted = extract_oblique_condition_nnz(condition)
        if extracted is not None:
            nnz, source = extracted
            observations.append(nnz)
            matching.append((source, repr(condition), nnz))

        stack.extend(child_objects(obj))

    return observations, matching


def tree_api_debug_lines(model):
    lines = [f"model_type={type(model).__name__}"]
    lines.append(f"model_fields={field_names(model)}")

    get_tree = getattr(model, "get_tree", None)
    if get_tree is None:
        lines.append("get_tree=unavailable")
        return lines

    try:
        tree = get_tree(0)
    except Exception as exc:
        lines.append(f"get_tree_error={type(exc).__name__}: {exc}")
        return lines

    lines.append(f"tree_type={type(tree).__name__}")
    lines.append(f"tree_fields={field_names(tree)}")
    root = getattr(tree, "root", None)
    if root is not None:
        lines.append(f"root_type={type(root).__name__}")
        lines.append(f"root_fields={field_names(root)}")
        condition = getattr(root, "condition", None)
        if condition is not None:
            lines.append(f"root_condition_type={type(condition).__name__}")
            lines.append(f"root_condition_fields={field_names(condition)}")
            lines.append(f"root_condition_repr={condition!r}")
    return lines


def extract_winning_projection_nnz(model):
    tree_observations, tree_matching = extract_tree_api_nnz(model)
    sources = []

    description = model_description(model)
    if description:
        sources.append(("describe()", description))

    sources.extend(read_serialized_text(model))

    observations = list(tree_observations)
    matching_lines = list(tree_matching)
    seen = set()
    for source_name, text in sources:
        for line in candidate_projection_lines(text):
            key = (source_name, line)
            if key in seen:
                continue
            seen.add(key)
            features = set(FEATURE_RE.findall(line))
            observations.append(len(features))
            matching_lines.append((source_name, line, len(features)))

    return observations, matching_lines, sources


def quantile(values, q):
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def format_number(value):
    if value is None:
        return "unavailable"
    return f"{value:g}"


def summarize_nnz(observations):
    if not observations:
        return {
            "n": 0,
            "mean": None,
            "var": None,
            "min": None,
            "q50": None,
            "q95": None,
            "max": None,
        }

    values = np.asarray(observations, dtype=np.float64)
    return {
        "n": len(observations),
        "mean": float(values.mean()),
        "var": float(values.var()),
        "min": float(values.min()),
        "q50": quantile(observations, 0.50),
        "q95": quantile(observations, 0.95),
        "max": float(values.max()),
    }


def print_case(name, args, intended_nnz, learner_args, summary, matching_lines, sources):
    print("=" * 88)
    print(f"CASE: {name}")
    print(f"n_features={args.n_features}")
    print(f"intended_e_nnz={intended_nnz}")
    print(f"learner_args={learner_args}")
    print(f"serialization_sources={[name for name, _ in sources]}")
    print(f"winning_projection_observations={len(matching_lines)}")
    print("TL;DR:")
    print(
        "  E[NNZ]: "
        f"intended={format_number(float(intended_nnz))}, "
        f"empirical_winning_mean={format_number(summary['mean'])}, "
        f"delta={format_number(None if summary['mean'] is None else summary['mean'] - intended_nnz)}"
    )
    print(
        "  winning_projection_nnz: "
        f"n={summary['n']}, "
        f"var={format_number(summary['var'])}, "
        f"min={format_number(summary['min'])}, "
        f"q50={format_number(summary['q50'])}, "
        f"q95={format_number(summary['q95'])}, "
        f"max={format_number(summary['max'])}"
    )

    if args.dump_matching_lines:
        print("MATCHING_PROJECTIONS:")
        for source_name, line, nnz in matching_lines[: args.max_dump_lines]:
            print(f"  [{source_name}] nnz={nnz}: {line}")
        if len(matching_lines) > args.max_dump_lines:
            print(f"  ... truncated {len(matching_lines) - args.max_dump_lines} lines")
    if args.dump_tree_api:
        print("TREE_API:")
        for line in args.tree_api_debug_lines:
            print(f"  {line}")
    print()


def case_model_dir(args, intended_nnz):
    if not args.save_model_dir:
        return None
    base = Path(args.save_model_dir)
    if len(args.intended_nnz) == 1:
        return base
    case_name = f"projection_density_factor_{intended_nnz:g}".replace(".", "p")
    return base / case_name


def run_case(args, intended_nnz):
    X, y = make_data(
        args.n_samples,
        args.n_features,
        args.signal_nnz,
        args.random_state,
    )
    train_ds = make_ydf_dict(X, y)
    learner_args = base_learner_args(args, intended_nnz)

    model = train_model(train_ds, learner_args, args.suppress_ydf_output)
    model_dir = case_model_dir(args, intended_nnz)
    if model_dir is not None:
        model_dir = save_model_for_inspection(model, model_dir)
        print_serialized_manifest(model_dir, args.serialized_preview_bytes)

    observations, matching_lines, sources = extract_winning_projection_nnz(model)
    summary = summarize_nnz(observations)
    args.tree_api_debug_lines = tree_api_debug_lines(model)
    print_case(
        name=f"projection_density_factor={intended_nnz}",
        args=args,
        intended_nnz=intended_nnz,
        learner_args=learner_args,
        summary=summary,
        matching_lines=matching_lines,
        sources=sources,
    )


def parse_projection_case(value):
    try:
        max_num_projections, exponent = value.split(":", 1)
        return int(max_num_projections), float(exponent)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "projection cases must have form MAX:EXPONENT, e.g. 64:0.5"
        ) from exc


def geometric_cdf(success_probability, trials):
    if trials <= 0:
        return 0.0
    if success_probability <= 0:
        return 0.0
    if success_probability >= 1:
        return 1.0
    return 1.0 - (1.0 - success_probability) ** trials


def geometric_quantile(success_probability, probability):
    if success_probability <= 0:
        return math.inf
    if success_probability >= 1:
        return 1
    return math.ceil(math.log1p(-probability) / math.log1p(-success_probability))


def modeled_projection_count(max_num_projections, exponent, n_features):
    return min(max_num_projections, int(n_features**exponent))


def projection_case_learner_args(args, max_num_projections, exponent):
    learner_args = base_learner_args(args, args.projection_test_nnz)
    learner_args |= {
        "max_depth": args.projection_test_max_depth,
        "sparse_oblique_max_num_projections": max_num_projections,
        "sparse_oblique_num_projections_exponent": exponent,
    }
    return learner_args


def projection_trial_success(args, seed, learner_args):
    X, y = make_data(
        args.projection_test_samples,
        args.n_features,
        args.projection_test_informative,
        seed,
    )
    trial_learner_args = dict(learner_args)
    trial_learner_args["random_seed"] = seed
    model = train_model(
        make_ydf_dict(X, y),
        trial_learner_args,
        args.suppress_ydf_output,
    )
    condition = root_condition(model)
    attributes = extract_oblique_condition_attributes(condition)
    if attributes is None:
        return {
            "success": False,
            "root_is_oblique": False,
            "root_nnz": None,
            "attributes": None,
        }

    # YDF's Python tree API reports column attributes as 1-based feature indices in
    # this build: a dataset with columns f0..f31 produces sparse-oblique attributes
    # in the range [1, 32]. The synthetic signal is deliberately placed in the first
    # k columns, so the corresponding informative attribute ids are 1..k.
    informative_attributes = set(range(1, args.projection_test_informative + 1))
    attribute_set = set(attributes)
    return {
        "success": informative_attributes.issubset(attribute_set),
        "root_is_oblique": True,
        "root_nnz": len(attributes),
        "attributes": attributes,
    }


def summarize_projection_outcomes(outcomes):
    successes = [outcome["success"] for outcome in outcomes]
    root_oblique = [outcome["root_is_oblique"] for outcome in outcomes]
    nnz = [
        outcome["root_nnz"]
        for outcome in outcomes
        if outcome["root_nnz"] is not None
    ]
    nnz_summary = summarize_nnz(nnz)
    return {
        "n": len(outcomes),
        "successes": int(sum(successes)),
        "success_rate": float(np.mean(successes)) if successes else None,
        "root_oblique_rate": float(np.mean(root_oblique)) if root_oblique else None,
        "root_nnz": nnz_summary,
    }


# Projection-count diagnostic theory
#
# We want to infer how many sparse-oblique projections YDF tries at a node. The
# tree model only stores the winning split, not the full candidate set, so this
# test turns candidate count into an observable success probability.
#
# Construct a synthetic binary problem with p total features and k informative
# features. The label is determined by an oblique sum of the first k coordinates.
# A projection is called a "success" if it contains all k informative dimensions.
# If YDF's sparse projection generator includes each feature independently with
# probability q, and we set sparse_oblique_projection_density_factor = E[NNZ],
# then:
#
#   q = E[NNZ] / p
#   s = P(one projection contains all k informative dims) = q ** k
#
# Let T be the number of projection rolls until the first successful projection.
# Then:
#
#   T ~ Geometric(s), support {1, 2, 3, ...}
#   P(T = t)  = (1 - s) ** (t - 1) * s
#   P(T <= m) = 1 - (1 - s) ** m
#   E[T]      = 1 / s
#   Var[T]    = (1 - s) / s ** 2
#
# Therefore, if a YDF setting causes m projections to be tried at the root, the
# chance that at least one candidate projection captures all informative
# dimensions is P(T <= m). Across many random seeds, the observed fraction of
# root winning projections containing all informative dimensions should track
# that CDF. This is not a pure generator audit: it assumes that when a successful
# candidate exists, the split search usually selects it because the synthetic
# target is explicitly aligned with those informative dimensions. For that
# reason, the default uses YDF max_depth=2, a low-dimensional signal, and many
# examples so root split quality dominates.
#
# With the defaults p=32, k=2, E[NNZ]=8:
#
#   q = 8 / 32 = 0.25
#   s = 0.25 ** 2 = 0.0625
#   E[T] = 16
#
# This gives a useful response curve:
#
#   m=1  -> P(T <= m) ~= 0.0625
#   m=5  -> P(T <= m) ~= 0.2758
#   m=10 -> P(T <= m) ~= 0.4755
#   m=32 -> P(T <= m) ~= 0.8732
#   m=64 -> P(T <= m) ~= 0.9840
#
# Those probabilities are far enough apart to distinguish common interpretations
# of sparse_oblique_max_num_projections and
# sparse_oblique_num_projections_exponent, e.g. whether exponent=0.0 means one
# projection, exponent=0.5 means roughly sqrt(p), and exponent=1.0 means p
# projections capped by max_num_projections.
def run_projection_count_trials(args):
    p = args.n_features
    k = args.projection_test_informative
    intended_nnz = args.projection_test_nnz
    inclusion_probability = intended_nnz / p
    single_projection_success = inclusion_probability**k
    expected_trials = 1.0 / single_projection_success
    variance_trials = (1.0 - single_projection_success) / (
        single_projection_success**2
    )

    print("=" * 88)
    print("PROJECTION COUNT GEOMETRIC MODEL")
    print(f"n_features={p}")
    print(f"n_informative={k}")
    print(f"intended_e_nnz={intended_nnz}")
    print(f"feature_inclusion_probability={inclusion_probability:g}")
    print(f"single_projection_success_probability={single_projection_success:g}")
    print(f"geometric_expected_trials={expected_trials:g}")
    print(f"geometric_variance_trials={variance_trials:g}")
    print(f"geometric_median_trials={geometric_quantile(single_projection_success, 0.5)}")
    print(f"geometric_q95_trials={geometric_quantile(single_projection_success, 0.95)}")
    print()

    for max_num_projections, exponent in args.projection_cases:
        modeled_m = modeled_projection_count(max_num_projections, exponent, p)
        modeled_success = geometric_cdf(single_projection_success, modeled_m)
        learner_args = projection_case_learner_args(
            args, max_num_projections, exponent
        )

        outcomes = []
        for i in range(args.projection_test_repeats):
            seed = args.random_state + i
            outcomes.append(projection_trial_success(args, seed, learner_args))
        summary = summarize_projection_outcomes(outcomes)
        observed = summary["success_rate"]

        print("=" * 88)
        print(f"CASE: max_num_projections={max_num_projections}, exponent={exponent}")
        print(f"learner_args={learner_args}")
        print(f"modeled_projection_count={modeled_m}")
        print(f"modeled_success_probability={modeled_success:g}")
        print("TL;DR:")
        print(
            "  root_all_informative_hit_rate: "
            f"observed={format_number(observed)}, "
            f"modeled={format_number(modeled_success)}, "
            f"delta={format_number(None if observed is None else observed - modeled_success)}, "
            f"successes={summary['successes']}/{summary['n']}"
        )
        print(
            "  root_projection_nnz: "
            f"mean={format_number(summary['root_nnz']['mean'])}, "
            f"var={format_number(summary['root_nnz']['var'])}, "
            f"min={format_number(summary['root_nnz']['min'])}, "
            f"q50={format_number(summary['root_nnz']['q50'])}, "
            f"max={format_number(summary['root_nnz']['max'])}"
        )
        print(
            "  root_oblique_rate: "
            f"{format_number(summary['root_oblique_rate'])}"
        )
        if args.dump_projection_trials:
            print("ROOT_PROJECTIONS:")
            for outcome in outcomes[: args.max_dump_lines]:
                print(
                    f"  success={outcome['success']} "
                    f"nnz={outcome['root_nnz']} "
                    f"attributes={outcome['attributes']}"
                )
            if len(outcomes) > args.max_dump_lines:
                print(f"  ... truncated {len(outcomes) - args.max_dump_lines} trials")
        print()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trial",
        choices=["all", "nnz", "projections"],
        default="all",
        help="Diagnostic trial group to run.",
    )
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--n-features", type=int, default=32)
    parser.add_argument("--signal-nnz", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=123)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-examples", type=int, default=2)
    parser.add_argument("--bootstrap-size-ratio", type=float, default=1.0)
    parser.add_argument("--num-projections", type=int, default=64)
    parser.add_argument("--num-projections-exponent", type=float, default=1.0)
    parser.add_argument(
        "--intended-nnz",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
        help="Values for sparse_oblique_projection_density_factor.",
    )
    parser.add_argument(
        "--dump-matching-lines",
        action="store_true",
        help="Print serialized lines used for NNZ counting.",
    )
    parser.add_argument(
        "--dump-tree-api",
        action="store_true",
        help="Print model/tree/root public fields for parser debugging.",
    )
    parser.add_argument("--max-dump-lines", type=int, default=40)
    parser.add_argument(
        "--save-model-dir",
        help="Save the first trained YDF model to this directory for inspection.",
    )
    parser.add_argument(
        "--serialized-preview-bytes",
        type=int,
        default=2048,
        help="Number of bytes to preview for each text-looking serialized file.",
    )
    parser.add_argument(
        "--show-ydf-output",
        dest="suppress_ydf_output",
        action="store_false",
        default=True,
        help="Show YDF's native training logs. Suppressed by default.",
    )
    parser.add_argument(
        "--projection-cases",
        type=parse_projection_case,
        nargs="+",
        default=[
            (1, 1.0),
            (5, 1.0),
            (10, 1.0),
            (64, 0.0),
            (64, 0.5),
            (64, 1.0),
        ],
        help=(
            "Projection-count cases as MAX:EXPONENT for YDF's "
            "sparse_oblique_max_num_projections and "
            "sparse_oblique_num_projections_exponent."
        ),
    )
    parser.add_argument("--projection-test-repeats", type=int, default=100)
    parser.add_argument("--projection-test-samples", type=int, default=4096)
    parser.add_argument("--projection-test-informative", type=int, default=2)
    parser.add_argument("--projection-test-nnz", type=float, default=8.0)
    parser.add_argument("--projection-test-max-depth", type=int, default=2)
    parser.add_argument(
        "--dump-projection-trials",
        action="store_true",
        help="Print root projection attributes for projection-count trials.",
    )

    args = parser.parse_args()
    if args.n_samples < 2:
        parser.error("--n-samples must be at least 2")
    if args.n_features < 2:
        parser.error("--n-features must be at least 2")
    if args.signal_nnz < 1:
        parser.error("--signal-nnz must be at least 1")
    if args.max_depth < 1:
        parser.error("--max-depth must be at least 1")
    if args.min_examples < 1:
        parser.error("--min-examples must be at least 1")
    if args.num_projections < 1:
        parser.error("--num-projections must be at least 1")
    if args.projection_test_repeats < 1:
        parser.error("--projection-test-repeats must be at least 1")
    if args.projection_test_samples < 2:
        parser.error("--projection-test-samples must be at least 2")
    if args.projection_test_informative < 1:
        parser.error("--projection-test-informative must be at least 1")
    if args.projection_test_informative > args.n_features:
        parser.error("--projection-test-informative cannot exceed --n-features")
    if args.projection_test_nnz <= 0:
        parser.error("--projection-test-nnz must be positive")
    if args.projection_test_nnz > args.n_features:
        parser.error("--projection-test-nnz cannot exceed --n-features")
    if args.projection_test_max_depth < 1:
        parser.error("--projection-test-max-depth must be at least 1")
    return args


def main():
    args = parse_args()
    if args.trial in {"all", "nnz"}:
        for intended_nnz in args.intended_nnz:
            run_case(args, intended_nnz)
    if args.trial in {"all", "projections"}:
        run_projection_count_trials(args)


if __name__ == "__main__":
    main()
