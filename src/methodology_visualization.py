import argparse
import csv
import json
import os
from pathlib import Path
import struct
import time

import numpy as np

from cuml.ensemble import SPORFClassifier
from cuml.testing.utils import get_handle


DEFAULT_OUTPUT_DIR = Path(".local/methodology-visualization")
CLASS_COLORS = ["#6CA6E8", "#FF9D3D", "#7BD66F", "#D982C1"]
NODE_COLORS = ["#F4C75C", "#9DDF7D", "#7FB3FF", "#D99BFF", "#F5DF6D", "#77D9C7", "#FF8F73"]
BG_COLOR = "#050608"
PANEL_FACE = "#080B10"
PANEL_EDGE = "#7C8796"
TEXT_COLOR = "#F3F4F6"
MUTED_TEXT = "#B9C0CA"
PROJECTION_COLOR = "#FF9D3D"
THRESHOLD_COLOR = "#FF4D43"
TESSERACT_EDGES = [
    (a, b)
    for a in range(16)
    for b in range(a + 1, 16)
    if bin(a ^ b).count("1") == 1
]
TESSERACT_VERTICES = np.asarray(
    [[(idx >> bit) & 1 for bit in range(4)] for idx in range(16)],
    dtype=np.float32,
)
CUBE_PROJECT_3D_TO_2D = np.asarray(
    [
        [1.00, 0.00],
        [0.34, 0.78],
        [-0.36, 0.50],
    ],
    dtype=np.float32,
)
TESSERACT_INNER_SCALE = 0.46


class BinaryReader:
    def __init__(self, payload):
        self.payload = memoryview(payload)
        self.offset = 0

    def read(self, fmt):
        size = struct.calcsize(fmt)
        if self.offset + size > len(self.payload):
            raise ValueError("Truncated SPORF serialization payload")
        value = struct.unpack_from(fmt, self.payload, self.offset)
        self.offset += size
        return value[0] if len(value) == 1 else value

    def read_vector(self, fmt):
        size = self.read("<Q")
        if size == 0:
            return []
        item_size = struct.calcsize(fmt)
        n_bytes = size * item_size
        if self.offset + n_bytes > len(self.payload):
            raise ValueError("Truncated SPORF vector payload")
        values = struct.unpack_from("<" + fmt[-1] * size, self.payload, self.offset)
        self.offset += n_bytes
        return list(values)


def make_unit_tesseract_data(n_points, random_state):
    if n_points % 4 != 0:
        raise ValueError("n_points must be divisible by 4 for balanced 4-class data")

    rng = np.random.default_rng(random_state)
    points_per_class = n_points // 4
    centers = np.asarray(
        [
            [0.18, 0.18, 0.18, 0.18],
            [0.82, 0.18, 0.82, 0.18],
            [0.18, 0.82, 0.18, 0.82],
            [0.82, 0.82, 0.82, 0.82],
        ],
        dtype=np.float32,
    )

    X_parts = []
    y_parts = []
    for class_id, center in enumerate(centers):
        jitter = rng.normal(loc=0.0, scale=0.085, size=(points_per_class, 4))
        X_class = np.clip(center + jitter, 0.0, 1.0)
        X_parts.append(X_class.astype(np.float32))
        y_parts.append(np.full(points_per_class, class_id, dtype=np.int32))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    order = rng.permutation(n_points)
    return np.ascontiguousarray(X[order]), np.ascontiguousarray(y[order])


def train_forest(X, y, args):
    handle, _streams = get_handle(True, n_streams=args.n_streams)
    clf = SPORFClassifier(
        n_estimators=args.n_trees,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_samples=args.max_samples,
        max_features=args.num_projections,
        density=args.expected_nnz,
        n_bins=args.n_bins,
        split_criterion=0,
        max_leaves=-1,
        random_state=args.random_state,
        n_streams=args.n_streams,
        handle=handle,
        verbose=False,
    )
    t0 = time.perf_counter()
    clf.fit(X, y)
    fit_time = time.perf_counter() - t0
    return clf, fit_time


def extract_forest_payload(clf):
    state = clf.__getstate__()
    if "sporf_forest_bytes" in state:
        return state["sporf_forest_bytes"], "<f"
    if "sporf_forest64_bytes" in state:
        return state["sporf_forest64_bytes"], "<d"
    raise ValueError("Fitted SPORF classifier did not expose serialized forest bytes")


def parse_serialized_forest(payload, float_fmt):
    reader = BinaryReader(payload)
    magic = reader.read("<I")
    version = reader.read("<I")
    if magic != 0x53504F52:
        raise ValueError("Invalid SPORF serialization magic")
    if version != 1:
        raise ValueError(f"Unsupported SPORF serialization version: {version}")

    trees = []
    n_trees = reader.read("<Q")
    for _ in range(n_trees):
        tree = {
            "treeid": reader.read("<i"),
            "depth": reader.read("<i"),
            "leaf_counter": reader.read("<i"),
            "train_time_ms": reader.read("<d"),
            "num_outputs": reader.read("<i"),
            "vector_leaf": reader.read_vector(float_fmt),
            "nodes": [],
            "projection_vectors": [],
            "projection_indptr_storage": [],
            "projection_indices_storage": [],
            "projection_coeffs_storage": [],
        }
        n_nodes = reader.read("<Q")
        for node_id in range(n_nodes):
            is_leaf = reader.read("<B") != 0
            node = {
                "node_id": node_id,
                "is_leaf": is_leaf,
                "colid": reader.read("<i"),
                "threshold": reader.read(float_fmt),
                "best_metric": reader.read(float_fmt),
                "left_child": reader.read("<q"),
                "instance_count": reader.read("<i"),
            }
            node["right_child"] = node["left_child"] + 1 if node["left_child"] >= 0 else -1
            tree["nodes"].append(node)

        n_proj_vectors = reader.read("<Q")
        for _ in range(n_proj_vectors):
            tree["projection_vectors"].append(
                {
                    "n_proj_components": reader.read("<i"),
                    "indptr_offset": reader.read("<Q"),
                    "indices_offset": reader.read("<Q"),
                    "coeffs_offset": reader.read("<Q"),
                }
            )
        tree["projection_indptr_storage"] = reader.read_vector("<i")
        tree["projection_indices_storage"] = reader.read_vector("<i")
        tree["projection_coeffs_storage"] = reader.read_vector(float_fmt)
        trees.append(tree)

    return trees


def node_projection_vector(tree, node_id, n_features):
    if node_id >= len(tree["projection_vectors"]):
        return np.zeros(n_features, dtype=np.float32)
    projection = tree["projection_vectors"][node_id]
    if projection["n_proj_components"] <= 0:
        return np.zeros(n_features, dtype=np.float32)

    indptr_offset = int(projection["indptr_offset"])
    indices_offset = int(projection["indices_offset"])
    coeffs_offset = int(projection["coeffs_offset"])
    indptr = tree["projection_indptr_storage"]
    if indptr_offset + 1 >= len(indptr):
        return np.zeros(n_features, dtype=np.float32)

    start = indices_offset + int(indptr[indptr_offset])
    end = indices_offset + int(indptr[indptr_offset + 1])
    vector = np.zeros(n_features, dtype=np.float32)
    for index, coeff in zip(
        tree["projection_indices_storage"][start:end],
        tree["projection_coeffs_storage"][coeffs_offset : coeffs_offset + (end - start)],
    ):
        if 0 <= index < n_features:
            vector[index] = coeff
    return vector


def route_points_through_tree(tree, X):
    node_points = {0: np.arange(X.shape[0], dtype=np.int32)}
    for node in tree["nodes"]:
        node_id = node["node_id"]
        indices = node_points.get(node_id)
        if indices is None or len(indices) == 0 or node["is_leaf"]:
            continue
        vector = node_projection_vector(tree, node_id, X.shape[1])
        if not np.any(vector):
            continue
        values = X[indices] @ vector
        left_mask = values <= node["threshold"]
        node_points[node["left_child"]] = indices[left_mask]
        node_points[node["right_child"]] = indices[~left_mask]
    return node_points


def predict_parsed_forest(trees, X):
    if not trees:
        return np.zeros(X.shape[0], dtype=np.int32)

    n_outputs = int(trees[0]["num_outputs"])
    scores = np.zeros((X.shape[0], n_outputs), dtype=np.float32)
    for row_id, row in enumerate(X):
        for tree in trees:
            node_id = 0
            while 0 <= node_id < len(tree["nodes"]):
                node = tree["nodes"][node_id]
                if node["is_leaf"]:
                    base = node_id * int(tree["num_outputs"])
                    leaf = tree["vector_leaf"][base : base + int(tree["num_outputs"])]
                    scores[row_id, : len(leaf)] += np.asarray(leaf, dtype=np.float32)
                    break
                vector = node_projection_vector(tree, node_id, X.shape[1])
                value = float(row @ vector)
                node_id = node["left_child"] if value <= node["threshold"] else node["right_child"]
    return np.argmax(scores, axis=1).astype(np.int32)


def node_depths(tree):
    depths = {}
    stack = [(0, 0)]
    while stack:
        node_id, depth = stack.pop()
        if node_id < 0 or node_id >= len(tree["nodes"]):
            continue
        depths[node_id] = depth
        node = tree["nodes"][node_id]
        if not node["is_leaf"]:
            stack.append((node["right_child"], depth + 1))
            stack.append((node["left_child"], depth + 1))
    return depths


def tree_layout(tree, left_x=0.16, right_x=0.72, top_y=0.865, bottom_y=0.105):
    depths = node_depths(tree)
    max_depth = max(depths.values()) if depths else 0
    leaf_order = []

    def visit(node_id):
        if node_id < 0 or node_id >= len(tree["nodes"]):
            return
        node = tree["nodes"][node_id]
        if node["is_leaf"]:
            leaf_order.append(node_id)
            return
        visit(node["left_child"])
        visit(node["right_child"])

    visit(0)
    if not leaf_order:
        leaf_order = [0]

    leaf_x = {
        node_id: left_x + idx * ((right_x - left_x) / max(1, len(leaf_order) - 1))
        for idx, node_id in enumerate(leaf_order)
    }
    positions = {}

    def place(node_id):
        if node_id < 0 or node_id >= len(tree["nodes"]):
            return 0.5
        node = tree["nodes"][node_id]
        if node["is_leaf"]:
            x_pos = leaf_x.get(node_id, 0.5)
        else:
            x_pos = 0.5 * (place(node["left_child"]) + place(node["right_child"]))
        depth = depths.get(node_id, 0)
        y_pos = top_y - depth * ((top_y - bottom_y) / max(1, max_depth))
        positions[node_id] = np.asarray([x_pos, y_pos])
        return x_pos

    place(0)
    return positions, depths, max_depth


def tesseract_project(points):
    points = np.asarray(points, dtype=np.float32)

    def project_raw(values):
        cube = values[:, :3] @ CUBE_PROJECT_3D_TO_2D
        cube_vertices = TESSERACT_VERTICES[:8, :3] @ CUBE_PROJECT_3D_TO_2D
        cube_center = 0.5 * (cube_vertices.min(axis=0) + cube_vertices.max(axis=0))
        w = values[:, 3:4]
        scale = 1.0 - (1.0 - TESSERACT_INNER_SCALE) * w
        return cube_center + (cube - cube_center) * scale

    projected = project_raw(points)
    projected_vertices = project_raw(TESSERACT_VERTICES)
    min_xy = projected_vertices.min(axis=0)
    max_xy = projected_vertices.max(axis=0)
    return (projected - min_xy) / np.maximum(max_xy - min_xy, 1e-6)


def draw_node_panel(
    ax,
    center,
    width,
    height,
    X,
    y,
    indices,
    node,
    vector,
    node_color,
):
    import matplotlib.patches as patches

    x0, y0 = center[0] - width / 2, center[1] - height / 2
    rect = patches.FancyBboxPatch(
        (x0, y0),
        width,
        height,
        boxstyle="round,pad=0.01,rounding_size=0.008",
        facecolor=PANEL_FACE,
        edgecolor=PANEL_EDGE,
        linewidth=0.75,
        alpha=0.98,
        zorder=2,
    )
    ax.add_patch(rect)

    def clipped_plot(*args, **kwargs):
        lines = ax.plot(*args, **kwargs)
        for line in lines:
            line.set_clip_path(rect)
        return lines

    def clipped_scatter(*args, **kwargs):
        artist = ax.scatter(*args, **kwargs)
        artist.set_clip_path(rect)
        return artist

    def place(p):
        tess_x0 = x0 + 0.08 * width
        tess_x1 = x0 + 0.62 * width
        tess_y0 = y0 + 0.29 * height
        tess_y1 = y0 + 0.91 * height
        return np.asarray(
            [
                tess_x0 + p[0] * (tess_x1 - tess_x0),
                tess_y0 + p[1] * (tess_y1 - tess_y0),
            ]
        )

    vertices_2d = tesseract_project(TESSERACT_VERTICES)
    vertices_screen = np.asarray([place(vertex) for vertex in vertices_2d])
    tesseract_min = vertices_screen.min(axis=0)
    tesseract_max = vertices_screen.max(axis=0)
    for a, b in TESSERACT_EDGES:
        pa, pb = vertices_screen[a], vertices_screen[b]
        a_w = int(TESSERACT_VERTICES[a, 3])
        b_w = int(TESSERACT_VERTICES[b, 3])
        if a_w != b_w:
            edge_alpha = 0.20
            edge_width = 0.42
        elif a_w == 0:
            edge_alpha = 0.34
            edge_width = 0.55
        else:
            edge_alpha = 0.45
            edge_width = 0.55
        clipped_plot(
            [pa[0], pb[0]],
            [pa[1], pb[1]],
            color="#E6E8EC",
            alpha=edge_alpha,
            linewidth=edge_width,
            zorder=3,
        )

    points_2d = tesseract_project(X[indices]) if len(indices) else np.empty((0, 2))
    for point_2d, class_id in zip(points_2d, y[indices]):
        p = place(point_2d)
        clipped_scatter(
            p[0],
            p[1],
            s=5.5,
            color=CLASS_COLORS[int(class_id) % len(CLASS_COLORS)],
            alpha=0.34,
            linewidth=0,
            zorder=4,
        )

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector_unit = vector / norm
        origin_4d = np.full(4, 0.5, dtype=np.float32)
        arrow_4d = ray_to_unit_box_boundary(origin_4d, vector_unit)
        back_4d = ray_to_unit_box_boundary(origin_4d, -vector_unit)
        origin_2d = place(tesseract_project(origin_4d[None, :])[0])
        arrow_2d = place(tesseract_project(arrow_4d[None, :])[0])
        back_2d = place(tesseract_project(back_4d[None, :])[0])
        clipped_plot(
            [back_2d[0], arrow_2d[0]],
            [back_2d[1], arrow_2d[1]],
            color=BG_COLOR,
            linewidth=1.75,
            alpha=0.80,
            zorder=5,
        )
        clipped_plot(
            [back_2d[0], arrow_2d[0]],
            [back_2d[1], arrow_2d[1]],
            color=PROJECTION_COLOR,
            linewidth=0.78,
            alpha=0.82,
            zorder=5,
        )
        ax.text(
            arrow_2d[0] + 0.006,
            arrow_2d[1] + 0.002,
            rf"$p_{{{node['node_id']}}}$",
            color=PROJECTION_COLOR,
            fontsize=7.8,
            fontstyle="italic",
            bbox={
                "boxstyle": "round,pad=0.06",
                "facecolor": BG_COLOR,
                "edgecolor": "none",
                "alpha": 0.72,
            },
            zorder=10,
        )

        if not node["is_leaf"]:
            if len(indices):
                projection_values = X[indices] @ vector
                threshold = float(node["threshold"])
                back_value = float(back_4d @ vector)
                arrow_value = float(arrow_4d @ vector)
                axis_span = max(arrow_value - back_value, 1e-6)
                axis_2d = arrow_2d - back_2d
                if np.linalg.norm(axis_2d) > 0:
                    axis_unit_2d = axis_2d / np.linalg.norm(axis_2d)
                    axis_perp_2d = np.asarray([-axis_unit_2d[1], axis_unit_2d[0]])
                else:
                    axis_unit_2d = np.asarray([1.0, 0.0])

                if abs(axis_unit_2d[1]) > abs(axis_unit_2d[0]):
                    gutter_min = np.asarray([x0 + 0.72 * width, y0 + 0.18 * height])
                    gutter_max = np.asarray([x0 + 0.93 * width, y0 + 0.82 * height])
                    axis_center = np.asarray([x0 + 0.825 * width, y0 + 0.52 * height])
                else:
                    gutter_min = np.asarray([x0 + 0.10 * width, y0 + 0.08 * height])
                    gutter_max = np.asarray([x0 + 0.82 * width, y0 + 0.26 * height])
                    axis_center = np.asarray([x0 + 0.46 * width, y0 + 0.17 * height])

                abs_axis = np.maximum(np.abs(axis_unit_2d), 1e-6)
                axis_perp_2d = np.asarray([-axis_unit_2d[1], axis_unit_2d[0]])
                gutter_size = gutter_max - gutter_min
                axis_length = 0.90 * min(gutter_size[0] / abs_axis[0], gutter_size[1] / abs_axis[1])
                display_back_2d = axis_center - 0.5 * axis_length * axis_unit_2d
                display_arrow_2d = axis_center + 0.5 * axis_length * axis_unit_2d
                display_axis_2d = display_arrow_2d - display_back_2d
                for tesseract_endpoint, gutter_endpoint in [
                    (back_2d, display_back_2d),
                    (arrow_2d, display_arrow_2d),
                ]:
                    clipped_plot(
                        [tesseract_endpoint[0], gutter_endpoint[0]],
                        [tesseract_endpoint[1], gutter_endpoint[1]],
                        color=MUTED_TEXT,
                        linewidth=0.58,
                        linestyle=(0, (1.0, 2.0)),
                        alpha=0.34,
                        zorder=5,
                    )
                clipped_scatter(
                    [display_back_2d[0], display_arrow_2d[0]],
                    [display_back_2d[1], display_arrow_2d[1]],
                    s=7,
                    color=PROJECTION_COLOR,
                    edgecolor=BG_COLOR,
                    linewidth=0.18,
                    alpha=0.70,
                    zorder=7,
                )
                clipped_plot(
                    [display_back_2d[0], display_arrow_2d[0]],
                    [display_back_2d[1], display_arrow_2d[1]],
                    color=PROJECTION_COLOR,
                    linewidth=0.65,
                    alpha=0.55,
                    zorder=6,
                )
                endpoint_label_offset = axis_perp_2d * 0.020 * min(width, height)
                axis_end_nudge = display_axis_2d / max(np.linalg.norm(display_axis_2d), 1e-6)
                ax.text(
                    display_back_2d[0] + endpoint_label_offset[0] - axis_end_nudge[0] * 0.008 * width,
                    display_back_2d[1] + endpoint_label_offset[1] - axis_end_nudge[1] * 0.008 * width,
                    "-",
                    color=MUTED_TEXT,
                    fontsize=5.8,
                    ha="center",
                    va="center",
                    zorder=8,
                )
                ax.text(
                    display_arrow_2d[0] + endpoint_label_offset[0] + axis_end_nudge[0] * 0.008 * width,
                    display_arrow_2d[1] + endpoint_label_offset[1] + axis_end_nudge[1] * 0.008 * width,
                    "+",
                    color=MUTED_TEXT,
                    fontsize=5.8,
                    ha="center",
                    va="center",
                    zorder=8,
                )

                threshold_t = np.clip((threshold - back_value) / axis_span, 0.0, 1.0)
                threshold_2d = display_back_2d + threshold_t * display_axis_2d
                tick_half = 0.018 * min(width, height)
                clipped_plot(
                    [
                        threshold_2d[0] - axis_perp_2d[0] * tick_half,
                        threshold_2d[0] + axis_perp_2d[0] * tick_half,
                    ],
                    [
                        threshold_2d[1] - axis_perp_2d[1] * tick_half,
                        threshold_2d[1] + axis_perp_2d[1] * tick_half,
                    ],
                    color=THRESHOLD_COLOR,
                    linewidth=1.45,
                    zorder=8,
                )
                for point_offset, (value, class_id) in enumerate(zip(projection_values, y[indices])):
                    t = np.clip((float(value) - back_value) / axis_span, 0.0, 1.0)
                    point_2d = display_back_2d + t * display_axis_2d
                    jitter = ((point_offset % 5) - 2) * 0.0055 * min(width, height)
                    point_2d = point_2d + axis_perp_2d * jitter
                    clipped_scatter(
                        point_2d[0],
                        point_2d[1],
                        s=11,
                        color=CLASS_COLORS[int(class_id) % len(CLASS_COLORS)],
                        edgecolor=BG_COLOR,
                        linewidth=0.15,
                        alpha=0.88,
                        zorder=7,
                    )

    label = f"n{node['node_id']}"
    if node["is_leaf"]:
        label += " leaf"
    ax.text(
        x0 + 0.025 * width,
        y0 + height - 0.025 * height,
        label,
        fontsize=5.9,
        ha="left",
        va="top",
        color=node_color,
        bbox={
            "boxstyle": "round,pad=0.10",
            "facecolor": BG_COLOR,
            "edgecolor": "none",
            "alpha": 0.72,
        },
        zorder=8,
    )


def draw_leaf_panel(ax, center, width, height, tree, node, indices, y, node_color):
    import matplotlib.patches as patches

    x0, y0 = center[0] - width / 2, center[1] - height / 2
    rect = patches.FancyBboxPatch(
        (x0, y0),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.008",
        facecolor=PANEL_FACE,
        edgecolor=PANEL_EDGE,
        linewidth=0.7,
        alpha=0.98,
        zorder=2,
    )
    ax.add_patch(rect)

    base = node["node_id"] * int(tree["num_outputs"])
    leaf = tree["vector_leaf"][base : base + int(tree["num_outputs"])]
    predicted = int(np.argmax(leaf)) if leaf else -1
    leaf_header_color = CLASS_COLORS[predicted % len(CLASS_COLORS)] if predicted >= 0 else node_color
    ax.text(
        center[0],
        y0 + height * 0.76,
        f"Leaf {node['node_id']}",
        color=leaf_header_color,
        fontsize=6.3,
        ha="center",
        va="center",
    )

    if len(indices):
        dot_step_x = min(width * 0.12, width * 0.78 / max(1, len(indices) - 1))
        dot_size = max(4.5, min(10.0, 0.58 * dot_step_x * 1000.0))
        start_x = center[0] - 0.5 * dot_step_x * (len(indices) - 1)
        dot_y = y0 + height * 0.50
        for offset, point_idx in enumerate(indices):
            ax.scatter(
                start_x + offset * dot_step_x,
                dot_y,
                s=dot_size,
                color=CLASS_COLORS[int(y[point_idx]) % len(CLASS_COLORS)],
                edgecolor=BG_COLOR,
                linewidth=0.25,
                alpha=0.95,
                zorder=4,
            )

    ax.text(
        center[0],
        y0 + height * 0.30,
        f"class {predicted}",
        color=TEXT_COLOR,
        fontsize=5.2,
        ha="center",
        va="center",
    )
    ax.text(
        center[0],
        y0 + height * 0.13,
        f"n = {len(indices)}",
        color=MUTED_TEXT,
        fontsize=5.0,
        ha="center",
        va="center",
    )


def format_sparse_vector(vector):
    parts = []
    for idx, value in enumerate(vector):
        if abs(float(value)) > 1e-6:
            parts.append(f"x{idx + 1}:{float(value):+.2f}")
    return "(" + ", ".join(parts or ["0"]) + ")"


def ray_to_unit_box_boundary(origin, direction):
    steps = np.full_like(direction, np.inf, dtype=np.float32)
    positive = direction > 1e-6
    negative = direction < -1e-6
    steps[positive] = (1.0 - origin[positive]) / direction[positive]
    steps[negative] = (0.0 - origin[negative]) / direction[negative]
    step = float(np.min(steps))
    if not np.isfinite(step) or step <= 0:
        return origin
    return np.clip(origin + step * direction, 0.0, 1.0)


def render_forest(path, trees, X, y, random_state, n_rows=None, n_cols=None, tree_index_offset=0):
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    rng = np.random.default_rng(random_state)
    subplot_left = 0.02
    subplot_right = 0.985
    subplot_top = 0.955
    subplot_bottom = 0.035
    subplot_hspace = 0.13
    subplot_wspace = 0.03
    if n_rows is None or n_cols is None:
        n_rows = len(trees)
        n_cols = 1
    if n_rows * n_cols < len(trees):
        raise ValueError("Grid shape is too small for the number of trees")
    fig_width = max(16, 7.2 * n_cols)
    fig_height = max(8, 5.8 * n_rows)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        constrained_layout=False,
    )
    fig.patch.set_facecolor(BG_COLOR)
    axes = np.asarray(axes, dtype=object).reshape(-1)
    for ax in axes[len(trees) :]:
        ax.set_facecolor(BG_COLOR)
        ax.axis("off")

    for tree_index, (ax, tree) in enumerate(zip(axes[: len(trees)], trees)):
        ax.set_facecolor(BG_COLOR)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(
            0.01,
            0.985,
            f"Tree {tree_index + 1 + tree_index_offset}",
            fontsize=10.5,
            weight="bold",
            color=TEXT_COLOR,
            ha="left",
            va="top",
        )
        show_side_panels = n_cols <= 2
        show_readout_panel = show_side_panels and (len(trees) == 1 or tree_index == 0)

        if len(trees) == 1:
            positions, depths, max_depth = tree_layout(tree, left_x=0.235, right_x=0.855)
        elif n_cols > 2:
            positions, depths, max_depth = tree_layout(tree, left_x=0.17, right_x=0.83)
        else:
            positions, depths, max_depth = tree_layout(tree)
        node_points = route_points_through_tree(tree, X)
        depth_groups = {}
        for node_id, depth in depths.items():
            depth_groups.setdefault(depth, []).append(node_id)

        for node in tree["nodes"]:
            if node["node_id"] not in positions or node["is_leaf"]:
                continue
            parent = positions[node["node_id"]]
            for child_key in ["left_child", "right_child"]:
                child = node[child_key]
                if child in positions:
                    child_pos = positions[child]
                    ax.plot(
                        [parent[0], child_pos[0]],
                        [parent[1] - 0.045, child_pos[1] + 0.045],
                        color="#C7CBD1",
                        linewidth=0.8,
                        alpha=0.70,
                        zorder=1,
                    )
                    label = (
                        f"<= {node['threshold']:.2f}"
                        if child_key == "left_child"
                        else f"> {node['threshold']:.2f}"
                    )
                    midpoint = 0.5 * (parent + child_pos)
                    ax.text(
                        midpoint[0],
                        midpoint[1] + 0.015,
                        label,
                        color=MUTED_TEXT,
                        fontsize=5.5,
                        ha="center",
                        va="center",
                        bbox={
                            "boxstyle": "round,pad=0.08",
                            "facecolor": BG_COLOR,
                            "edgecolor": "none",
                            "alpha": 0.86,
                        },
                        zorder=3,
                    )

        widest_depth = max(len(v) for v in depth_groups.values()) if depth_groups else 1
        row_height = fig_height * (subplot_top - subplot_bottom) / max(1, n_rows)
        row_width = fig_width * (subplot_right - subplot_left) / max(1, n_cols)
        row_aspect = row_width / max(row_height, 1e-6)
        panel_height = min(0.255, 0.82 / max(1, widest_depth))
        panel_width = min(0.25, panel_height * 1.18 / max(row_aspect, 1e-6))
        for node in tree["nodes"]:
            node_id = node["node_id"]
            if node_id not in positions:
                continue
            indices = node_points.get(node_id, np.asarray([], dtype=np.int32))
            node_color = NODE_COLORS[node_id % len(NODE_COLORS)]
            if node["is_leaf"]:
                draw_leaf_panel(
                    ax,
                    positions[node_id],
                    width=min(0.105, panel_width * 1.18),
                    height=0.072,
                    tree=tree,
                    node=node,
                    indices=indices,
                    y=y,
                    node_color=node_color,
                )
            else:
                vector = node_projection_vector(tree, node_id, X.shape[1])
                draw_node_panel(
                    ax,
                    positions[node_id],
                    panel_width,
                    panel_height,
                    X,
                    y,
                    indices,
                    node,
                    vector,
                    node_color,
                )

        if show_side_panels:
            side_x = 0.015 if len(trees) == 1 else 0.82
            class_dot_x = side_x + 0.008
            class_text_x = side_x + 0.03
            note_y = 0.92 if len(trees) == 1 else 0.96
            readout_y = 0.825 if len(trees) == 1 else 0.83
            legend_y = 0.72 if len(trees) == 1 else (0.69 if show_readout_panel else 0.88)
            vector_y = 0.52 if len(trees) == 1 else (0.48 if show_readout_panel else 0.67)
            if show_readout_panel:
                ax.text(
                    side_x,
                    note_y,
                    "Each internal node defines a sparse\n"
                    "one-dimensional view of the 4D unit\n"
                    "tesseract and splits by thresholding\n"
                    "the projected value.",
                    fontsize=6.8 if n_cols <= 2 else 5.7,
                    va="top",
                    ha="left",
                    color=MUTED_TEXT,
                    linespacing=1.22,
                    bbox={
                        "boxstyle": "round,pad=0.42",
                        "facecolor": PANEL_FACE,
                        "edgecolor": PANEL_EDGE,
                        "linewidth": 0.55,
                        "alpha": 0.92,
                    },
                )
                ax.text(
                    side_x,
                    readout_y,
                    "How to read a node\n"
                    "orange: projection direction p_i\n"
                    "on-axis dots: projected values\n"
                    "red tick: threshold value",
                    fontsize=7.5 if n_cols <= 2 else 6.4,
                    va="top",
                    ha="left",
                    color=MUTED_TEXT,
                    linespacing=1.25,
                    bbox={
                        "boxstyle": "round,pad=0.42",
                        "facecolor": PANEL_FACE,
                        "edgecolor": PANEL_EDGE,
                        "linewidth": 0.65,
                        "alpha": 0.96,
                    },
                )
            internal_nodes = [node for node in tree["nodes"] if not node["is_leaf"]]
            vector_lines = []
            for node in internal_nodes[:6]:
                vector = node_projection_vector(tree, node["node_id"], X.shape[1])
                color = NODE_COLORS[node["node_id"] % len(NODE_COLORS)]
                vector_lines.append((color, f"n{node['node_id']}: p_{node['node_id']}={format_sparse_vector(vector)}"))
            ax.text(
                side_x,
                legend_y,
                "Classes",
                fontsize=7.2 if n_cols <= 2 else 6.2,
                color=TEXT_COLOR,
                ha="left",
                va="top",
            )
            legend_y -= 0.038
            for class_id, color in enumerate(CLASS_COLORS):
                ax.scatter(
                    class_dot_x,
                    legend_y + 0.004,
                    s=24 if n_cols <= 2 else 16,
                    color=color,
                    edgecolor="none",
                    zorder=5,
                )
                ax.text(
                    class_text_x,
                    legend_y,
                    f"class {class_id}",
                    fontsize=6.5 if n_cols <= 2 else 5.3,
                    color=MUTED_TEXT,
                    ha="left",
                    va="center",
                )
                legend_y -= 0.034

            y_cursor = vector_y
            ax.text(
                side_x,
                y_cursor,
                "Sparse projection vectors",
                fontsize=7.6 if n_cols <= 2 else 6.6,
                color=TEXT_COLOR,
                ha="left",
                va="top",
            )
            y_cursor -= 0.055
            for color, line in vector_lines:
                ax.text(
                    side_x,
                    y_cursor,
                    line,
                    fontsize=6.2 if n_cols <= 2 else 5.1,
                    color=color,
                    ha="left",
                    va="top",
                    family="monospace",
                )
                y_cursor -= 0.042

    fig.suptitle(
        "SPORF Methodology: Partitioning High-Dimensional Data with Sparse Projection Trees",
        fontsize=15,
        weight="bold",
        color=TEXT_COLOR,
        x=0.015,
        y=0.993,
        ha="left",
    )
    fig.subplots_adjust(
        left=subplot_left,
        right=subplot_right,
        top=subplot_top,
        bottom=subplot_bottom,
        hspace=subplot_hspace,
        wspace=subplot_wspace,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_points(path, X, y):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["point_id", "class_id", "x0", "x1", "x2", "x3"])
        for idx, (point, class_id) in enumerate(zip(X, y)):
            writer.writerow([idx, int(class_id), *[float(value) for value in point]])
    return path


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(text)
    return path


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def read_points(path):
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    X = np.asarray(
        [[float(row[f"x{feature_idx}"]) for feature_idx in range(4)] for row in rows],
        dtype=np.float32,
    )
    y = np.asarray([int(row["class_id"]) for row in rows], dtype=np.int32)
    return X, y


def read_json(path):
    with path.open() as f:
        return json.load(f)


def render_outputs(output_dir, trees, X, y, args):
    png_1x4_path = args.output_png_1x4 or output_dir / "methodology_forest_1x4.png"
    png_2x2_path = args.output_png_2x2 or args.output_png or output_dir / "methodology_forest_2x2.png"
    rendered_1x4_path = render_forest(
        png_1x4_path,
        trees,
        X,
        y,
        args.random_state + 1,
        n_rows=1,
        n_cols=len(trees),
    )
    rendered_2x2_path = render_forest(
        png_2x2_path, trees, X, y, args.random_state + 1, n_rows=2, n_cols=2
    )
    rendered_tree_paths = []
    for tree_index, tree in enumerate(trees, start=1):
        rendered_tree_paths.append(
            render_forest(
                output_dir / f"methodology_tree_{tree_index}.png",
                [tree],
                X,
                y,
                args.random_state + 1,
                n_rows=1,
                n_cols=1,
                tree_index_offset=tree_index - 1,
            )
        )
    return rendered_1x4_path, rendered_2x2_path, rendered_tree_paths


def summarize(trees, X, y, fit_time, args):
    pred = predict_parsed_forest(trees, X)
    accuracy = float(np.mean(np.asarray(pred) == y))
    return {
        "n_points": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(np.unique(y))),
        "class_counts": {
            str(class_id): int(np.sum(y == class_id)) for class_id in np.unique(y)
        },
        "fit_time_seconds": fit_time,
        "training_accuracy": accuracy,
        "hyperparameters": {
            "n_estimators": args.n_trees,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "max_samples": args.max_samples,
            "max_features": args.num_projections,
            "density": args.expected_nnz,
            "n_bins": args.n_bins,
            "n_streams": args.n_streams,
            "random_state": args.random_state,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a tiny 4-class distribution in the unit tesseract and "
            "train a small cuML SPORF forest for methodology visualization."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-png",
        type=Path,
        default=None,
        help="Legacy 2x2 PNG output path override.",
    )
    parser.add_argument(
        "--output-png-1x4",
        type=Path,
        default=None,
        help="1x4 PNG output path. Defaults to <output-dir>/methodology_forest_1x4.png.",
    )
    parser.add_argument(
        "--output-png-2x2",
        type=Path,
        default=None,
        help="2x2 PNG output path. Defaults to <output-dir>/methodology_forest_2x2.png.",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help=(
            "Regenerate PNG renders from <output-dir>/points.csv and "
            "<output-dir>/forest_structure.json without retraining."
        ),
    )
    parser.add_argument("--n-points", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=20260807)
    parser.add_argument("--n-trees", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--max-samples", type=float, default=1.0)
    parser.add_argument("--num-projections", type=int, default=3)
    parser.add_argument("--expected-nnz", type=int, default=2)
    parser.add_argument("--n-bins", type=int, default=16)
    parser.add_argument("--n-streams", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir
    diagnostics_path = None
    if args.render_only:
        points_path = output_dir / "points.csv"
        forest_path = output_dir / "forest_structure.json"
        X, y = read_points(points_path)
        trees = read_json(forest_path)
        summary_path = output_dir / "summary.json"
        summary = read_json(summary_path) if summary_path.exists() else None
    else:
        X, y = make_unit_tesseract_data(args.n_points, args.random_state)
        clf, fit_time = train_forest(X, y, args)
        forest_payload, float_fmt = extract_forest_payload(clf)
        trees = parse_serialized_forest(forest_payload, float_fmt)
        summary = summarize(trees, X, y, fit_time, args)
        summary["rendered_tree_count"] = len(trees)
        points_path = write_points(output_dir / "points.csv", X, y)
        summary_path = write_json(output_dir / "summary.json", summary)
        forest_path = write_json(output_dir / "forest_structure.json", trees)
        if hasattr(clf, "get_diagnostics_csv"):
            diagnostics_path = write_text(
                output_dir / "tree_diagnostics.csv",
                clf.get_diagnostics_csv(),
            )

    rendered_1x4_path, rendered_2x2_path, rendered_tree_paths = render_outputs(
        output_dir, trees, X, y, args
    )

    if args.render_only:
        print(f"Loaded points: {points_path}")
        print(f"Loaded parsed forest: {forest_path}")
        if summary is not None:
            print(f"Loaded summary: {summary_path}")
    else:
        print(f"Generated {summary['n_points']} points in the unit tesseract")
        print(f"Trained {args.n_trees} SPORF trees in {summary['fit_time_seconds']:.4f}s")
        print(f"Training accuracy: {summary['training_accuracy']:.4f}")
        print(f"Wrote points: {points_path}")
        print(f"Wrote summary: {summary_path}")
        print(f"Wrote parsed forest: {forest_path}")
    print(f"Wrote methodology visualization 1x4: {rendered_1x4_path}")
    print(f"Wrote methodology visualization 2x2: {rendered_2x2_path}")
    for tree_path in rendered_tree_paths:
        print(f"Wrote single-tree visualization: {tree_path}")
    if diagnostics_path is not None:
        print(f"Wrote tree diagnostics: {diagnostics_path}")


if __name__ == "__main__":
    main()
