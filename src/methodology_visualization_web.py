import argparse
import json
from pathlib import Path

import numpy as np

from methodology_visualization import (
    BG_COLOR,
    CLASS_COLORS,
    MUTED_TEXT,
    NODE_COLORS,
    PANEL_EDGE,
    PANEL_FACE,
    PROJECTION_COLOR,
    TESSERACT_EDGES,
    TESSERACT_VERTICES,
    TEXT_COLOR,
    THRESHOLD_COLOR,
    format_sparse_vector,
    node_projection_vector,
    ray_to_unit_box_boundary,
    read_json,
    read_points,
    route_points_through_tree,
    tesseract_project,
    tree_layout,
)


DEFAULT_OUTPUT_DIR = Path(".local/methodology-visualization")
TESSERACT_VIEW_DEPTH_3D = np.asarray([-0.22, -0.34, 0.92], dtype=np.float32)
TESSERACT_VIEW_DEPTH_3D /= np.linalg.norm(TESSERACT_VIEW_DEPTH_3D)


def tesseract_view_depth(points):
    points = np.asarray(points, dtype=np.float32)
    cube_depth = points[:, :3] @ TESSERACT_VIEW_DEPTH_3D
    # The w=1 cube is drawn as the smaller, inset cube, i.e. visually farther away.
    return cube_depth - 0.55 * points[:, 3]


def as_xy(point):
    return {"x": float(point[0]), "y": float(point[1])}


def transform_point(tile, point):
    return np.asarray(
        [
            tile["x"] + float(point[0]) * tile["width"],
            tile["y"] + (1.0 - float(point[1])) * tile["height"],
        ],
        dtype=np.float32,
    )


def card_y(y0, height, fraction_from_bottom):
    return y0 + (1.0 - float(fraction_from_bottom)) * height


def class_counts(y, indices):
    return {
        str(class_id): int(np.sum(y[indices] == class_id))
        for class_id in sorted(np.unique(y[indices]).tolist())
    }


def leaf_prediction(tree, node):
    base = node["node_id"] * int(tree["num_outputs"])
    leaf = tree["vector_leaf"][base : base + int(tree["num_outputs"])]
    return int(np.argmax(leaf)) if leaf else -1


def tesseract_box(x0, y0, width, height, gutter_orientation):
    if gutter_orientation == "right":
        return {
            "x0": x0 + 0.065 * width,
            "x1": x0 + 0.625 * width,
            "y0": 0.185,
            "y1": 0.825,
        }
    return {
        "x0": x0 + 0.165 * width,
        "x1": x0 + 0.755 * width,
        "y0": 0.335,
        "y1": 0.92,
    }


def card_place(x0, y0, width, height, point, gutter_orientation="bottom"):
    box = tesseract_box(x0, y0, width, height, gutter_orientation)
    tess_left = box["x0"]
    tess_right = box["x1"]
    tess_top = card_y(y0, height, box["y1"])
    tess_bottom = card_y(y0, height, box["y0"])
    scale = min(tess_right - tess_left, tess_bottom - tess_top)
    center_x = 0.5 * (tess_left + tess_right)
    center_y = 0.5 * (tess_top + tess_bottom)
    return np.asarray(
        [
            center_x + (float(point[0]) - 0.5) * scale,
            center_y - (float(point[1]) - 0.5) * scale,
        ],
        dtype=np.float32,
    )


def rect_boundary_point(center, direction, width, height):
    direction = np.asarray(direction, dtype=np.float32)
    if np.linalg.norm(direction) < 1e-6:
        return np.asarray(center, dtype=np.float32)
    half_width = 0.5 * width
    half_height = 0.5 * height
    scale_x = half_width / max(abs(float(direction[0])), 1e-6)
    scale_y = half_height / max(abs(float(direction[1])), 1e-6)
    return np.asarray(center, dtype=np.float32) + direction * min(scale_x, scale_y)


def build_internal_node_card(tree, node, X, y, indices, center, width, height, color):
    x0, y0 = center[0] - width / 2, center[1] - height / 2
    vector = node_projection_vector(tree, node["node_id"], X.shape[1])
    gutter_orientation = "bottom"
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector_unit = vector / norm
        origin_4d = np.full(4, 0.5, dtype=np.float32)
        arrow_4d = ray_to_unit_box_boundary(origin_4d, vector_unit)
        back_4d = ray_to_unit_box_boundary(origin_4d, -vector_unit)
        preview_arrow = tesseract_project(arrow_4d[None, :])[0]
        preview_back = tesseract_project(back_4d[None, :])[0]
        preview_axis = preview_arrow - preview_back
        if abs(preview_axis[1]) > abs(preview_axis[0]):
            gutter_orientation = "right"

    vertices_2d = tesseract_project(TESSERACT_VERTICES)
    vertices_screen = np.asarray(
        [
            card_place(x0, y0, width, height, vertex, gutter_orientation)
            for vertex in vertices_2d
        ],
        dtype=np.float32,
    )
    tesseract_min = vertices_screen.min(axis=0)
    tesseract_max = vertices_screen.max(axis=0)

    edges = []
    vertex_depth = tesseract_view_depth(TESSERACT_VERTICES)
    vertex_depth = (vertex_depth - vertex_depth.min()) / max(
        float(vertex_depth.max() - vertex_depth.min()), 1e-6
    )
    for edge_id, (a, b) in enumerate(TESSERACT_EDGES):
        a_w = int(TESSERACT_VERTICES[a, 3])
        b_w = int(TESSERACT_VERTICES[b, 3])
        if a_w != b_w:
            edge_kind = "cross"
        elif a_w == 0:
            edge_kind = "outer"
        else:
            edge_kind = "inner"
        edges.append(
            {
                "id": edge_id,
                "kind": edge_kind,
                "depth_a": float(vertex_depth[a]),
                "depth_b": float(vertex_depth[b]),
                "a": as_xy(vertices_screen[a]),
                "b": as_xy(vertices_screen[b]),
            }
        )

    points_2d = tesseract_project(X[indices]) if len(indices) else np.empty((0, 2))
    points = [
        {
            "x": float(card_place(x0, y0, width, height, point, gutter_orientation)[0]),
            "y": float(card_place(x0, y0, width, height, point, gutter_orientation)[1]),
            "class_id": int(class_id),
        }
        for point, class_id in zip(points_2d, y[indices])
    ]

    projection = None
    if norm > 0 and len(indices):
        vector_unit = vector / norm
        origin_4d = np.full(4, 0.5, dtype=np.float32)
        arrow_4d = ray_to_unit_box_boundary(origin_4d, vector_unit)
        back_4d = ray_to_unit_box_boundary(origin_4d, -vector_unit)
        arrow_2d = card_place(
            x0,
            y0,
            width,
            height,
            tesseract_project(arrow_4d[None, :])[0],
            gutter_orientation,
        )
        back_2d = card_place(
            x0,
            y0,
            width,
            height,
            tesseract_project(back_4d[None, :])[0],
            gutter_orientation,
        )

        threshold = float(node["threshold"])
        axis_2d = arrow_2d - back_2d
        if np.linalg.norm(axis_2d) > 0:
            axis_unit_2d = axis_2d / np.linalg.norm(axis_2d)
        else:
            axis_unit_2d = np.asarray([1.0, 0.0], dtype=np.float32)
        axis_perp_2d = np.asarray([-axis_unit_2d[1], axis_unit_2d[0]], dtype=np.float32)

        if abs(axis_unit_2d[1]) > abs(axis_unit_2d[0]):
            gutter_min = np.asarray([x0 + 0.74 * width, card_y(y0, height, 0.78)])
            gutter_max = np.asarray([x0 + 0.90 * width, card_y(y0, height, 0.22)])
            axis_center = np.asarray([x0 + 0.82 * width, card_y(y0, height, 0.50)])
        else:
            gutter_min = np.asarray([x0 + 0.14 * width, card_y(y0, height, 0.24)])
            gutter_max = np.asarray([x0 + 0.78 * width, card_y(y0, height, 0.10)])
            axis_center = np.asarray([x0 + 0.46 * width, card_y(y0, height, 0.17)])

        abs_axis = np.maximum(np.abs(axis_unit_2d), 1e-6)
        gutter_size = gutter_max - gutter_min
        axis_length = 0.90 * min(gutter_size[0] / abs_axis[0], gutter_size[1] / abs_axis[1])
        display_back_2d = axis_center - 0.5 * axis_length * axis_unit_2d
        display_arrow_2d = axis_center + 0.5 * axis_length * axis_unit_2d
        display_axis_2d = display_arrow_2d - display_back_2d
        display_axis_norm = max(np.linalg.norm(display_axis_2d), 1e-6)
        display_axis_unit_2d = display_axis_2d / display_axis_norm
        display_axis_perp_2d = np.asarray(
            [-display_axis_unit_2d[1], display_axis_unit_2d[0]],
            dtype=np.float32,
        )

        projection_values = X[indices] @ vector
        value_min = float(np.min(projection_values))
        value_max = float(np.max(projection_values))
        min_point_idx = int(np.argmin(projection_values))
        max_point_idx = int(np.argmax(projection_values))
        extrema_points_2d = tesseract_project(
            X[indices[[min_point_idx, max_point_idx]]]
        )
        min_point_2d = card_place(
            x0,
            y0,
            width,
            height,
            extrema_points_2d[0],
            gutter_orientation,
        )
        max_point_2d = card_place(
            x0,
            y0,
            width,
            height,
            extrema_points_2d[1],
            gutter_orientation,
        )
        value_span = value_max - value_min
        if value_span < 1e-6:
            value_pad = max(1.0, abs(value_min)) * 0.5
            value_min -= value_pad
            value_max += value_pad
            value_span = value_max - value_min

        threshold_t = np.clip((threshold - value_min) / value_span, 0.0, 1.0)
        threshold_2d = display_back_2d + threshold_t * display_axis_2d
        tick_half = 0.018 * min(width, height)
        band_half_axis = 0.036 * min(width, height)
        band_half_perp = 0.150 * min(width, height)
        threshold_band = [
            threshold_2d
            - display_axis_unit_2d * band_half_axis
            - display_axis_perp_2d * band_half_perp,
            threshold_2d
            + display_axis_unit_2d * band_half_axis
            - display_axis_perp_2d * band_half_perp,
            threshold_2d
            + display_axis_unit_2d * band_half_axis
            + display_axis_perp_2d * band_half_perp,
            threshold_2d
            - display_axis_unit_2d * band_half_axis
            + display_axis_perp_2d * band_half_perp,
        ]
        projected_points = []
        for point_offset, (value, class_id) in enumerate(zip(projection_values, y[indices])):
            t = np.clip((float(value) - value_min) / value_span, 0.0, 1.0)
            point_2d = display_back_2d + t * display_axis_2d
            jitter = ((point_offset % 5) - 2) * 0.0055 * min(width, height)
            point_2d = point_2d + axis_perp_2d * jitter
            projected_points.append(
                {
                    "x": float(point_2d[0]),
                    "y": float(point_2d[1]),
                    "class_id": int(class_id),
                }
            )

        sign_offset = 0.084 * min(width, height)
        projection = {
            "vector": [float(v) for v in vector.tolist()],
            "label": f"p_{node['node_id']}",
            "label_position": as_xy(arrow_2d + np.asarray([0.006 * width, 0.002 * height])),
            "tesseract_axis": {"a": as_xy(back_2d), "b": as_xy(arrow_2d)},
            "gutter_axis": {"a": as_xy(display_back_2d), "b": as_xy(display_arrow_2d)},
            "connectors": [
                {"a": as_xy(min_point_2d), "b": as_xy(display_back_2d)},
                {"a": as_xy(max_point_2d), "b": as_xy(display_arrow_2d)},
            ],
            "threshold_tick": {
                "a": as_xy(threshold_2d - display_axis_perp_2d * tick_half),
                "b": as_xy(threshold_2d + display_axis_perp_2d * tick_half),
            },
            "threshold_band": [as_xy(point) for point in threshold_band],
            "projected_points": projected_points,
            "negative_label": as_xy(
                display_back_2d - display_axis_unit_2d * sign_offset
            ),
            "positive_label": as_xy(
                display_arrow_2d + display_axis_unit_2d * sign_offset
            ),
        }

    return {
        "type": "internal",
        "id": int(node["node_id"]),
        "color": color,
        "rect": {"x": float(x0), "y": float(y0), "width": float(width), "height": float(height)},
        "label": f"n{node['node_id']}",
        "tesseract": {"edges": edges, "points": points},
        "projection": projection,
        "class_counts": class_counts(y, indices) if len(indices) else {},
    }


def build_leaf_card(tree, node, y, indices, center, width, height, color):
    x0, y0 = center[0] - width / 2, center[1] - height / 2
    predicted = leaf_prediction(tree, node)
    dot_step_x = min(width * 0.12, width * 0.78 / max(1, len(indices) - 1)) if len(indices) else 0.0
    start_x = center[0] - 0.5 * dot_step_x * (len(indices) - 1) if len(indices) else center[0]
    dots = [
        {
            "x": float(start_x + offset * dot_step_x),
            "y": float(card_y(y0, height, 0.50)),
            "class_id": int(y[point_idx]),
        }
        for offset, point_idx in enumerate(indices)
    ]
    return {
        "type": "leaf",
        "id": int(node["node_id"]),
        "color": CLASS_COLORS[predicted % len(CLASS_COLORS)] if predicted >= 0 else color,
        "rect": {"x": float(x0), "y": float(y0), "width": float(width), "height": float(height)},
        "label": f"Leaf {node['node_id']}",
        "predicted_class": predicted,
        "n": int(len(indices)),
        "dots": dots,
        "class_counts": class_counts(y, indices) if len(indices) else {},
    }


def build_tree_payload(tree, tree_index, X, y, tile, layout_mode="standard"):
    if layout_mode == "banner":
        positions, depths, _max_depth = tree_layout(
            tree, left_x=0.08, right_x=0.92, top_y=0.86, bottom_y=0.10
        )
        banner_lower_level_shift = 0.05
        banner_y_by_depth = {0: 0.82, 1: 0.66, 2: 0.285}
        for node_id, depth in depths.items():
            if node_id in positions:
                node = tree["nodes"][node_id]
                if node["is_leaf"] and depth == 3:
                    positions[node_id][1] = -0.006 - banner_lower_level_shift
                elif depth == 1:
                    positions[node_id][0] = 0.5 + 0.84 * (positions[node_id][0] - 0.5)
                    positions[node_id][1] = (
                        banner_y_by_depth[depth] - banner_lower_level_shift
                    )
                elif depth >= 1:
                    positions[node_id][1] = (
                        banner_y_by_depth.get(depth, 0.06) - banner_lower_level_shift
                    )
                else:
                    positions[node_id][1] = banner_y_by_depth.get(depth, 0.06)
    else:
        positions, depths, _max_depth = tree_layout(tree)
    node_points = route_points_through_tree(tree, X)
    depth_groups = {}
    for node_id, depth in depths.items():
        depth_groups.setdefault(depth, []).append(node_id)
    widest_depth = max(len(v) for v in depth_groups.values()) if depth_groups else 1
    if layout_mode == "banner":
        card_height = min(304.0, tile["height"] * 1.27 / max(1, widest_depth))
        card_width = min(438.0, card_height * 1.40)
        leaf_width = min(230.0, card_width * 1.00)
        leaf_height = 62.0
    else:
        card_height = min(172.0, tile["height"] * 0.80 / max(1, widest_depth))
        card_width = min(240.0, card_height * 1.32)
        leaf_width = min(132.0, card_width * 1.18)
        leaf_height = 48.0

    absolute_positions = {
        node_id: transform_point(tile, position) for node_id, position in positions.items()
    }
    edges = []
    split_labels = []
    for node in tree["nodes"]:
        if node["node_id"] not in absolute_positions or node["is_leaf"]:
            continue
        parent = absolute_positions[node["node_id"]]
        for child_key, split_symbol in [("left_child", "<="), ("right_child", ">")]:
            child = node[child_key]
            if child not in absolute_positions:
                continue
            child_pos = absolute_positions[child]
            if layout_mode == "banner":
                child_is_leaf = bool(tree["nodes"][child]["is_leaf"])
                direction = child_pos - parent
                start = rect_boundary_point(parent, direction, card_width, card_height)
                end = rect_boundary_point(
                    child_pos,
                    -direction,
                    leaf_width if child_is_leaf else card_width,
                    leaf_height if child_is_leaf else card_height,
                )
            else:
                edge_gap = 0.045 * tile["height"]
                start = parent + np.asarray([0.0, edge_gap])
                end = child_pos + np.asarray([0.0, -edge_gap])
            midpoint = 0.5 * (parent + child_pos)
            edges.append({"a": as_xy(start), "b": as_xy(end)})
            split_text = (
                split_symbol
                if layout_mode == "banner"
                else f"{split_symbol} {node['threshold']:.2f}"
            )
            label_y = midpoint[1] + 0.015 * tile["height"]
            if layout_mode == "banner" and depths.get(node["node_id"]) == 2:
                parent_bottom = parent[1] + 0.5 * card_height
                child_top = child_pos[1] - 0.5 * leaf_height
                label_y = 0.5 * (parent_bottom + child_top)
            split_labels.append(
                {
                    "x": float(midpoint[0]),
                    "y": float(label_y),
                    "text": split_text,
                }
            )

    cards = []
    vector_lines = []
    for node in tree["nodes"]:
        node_id = node["node_id"]
        if node_id not in absolute_positions:
            continue
        indices = node_points.get(node_id, np.asarray([], dtype=np.int32))
        color = NODE_COLORS[node_id % len(NODE_COLORS)]
        if node["is_leaf"]:
            cards.append(
                build_leaf_card(
                    tree,
                    node,
                    y,
                    indices,
                    absolute_positions[node_id],
                    leaf_width,
                    leaf_height,
                    color,
                )
            )
        else:
            vector = node_projection_vector(tree, node_id, X.shape[1])
            cards.append(
                build_internal_node_card(
                    tree,
                    node,
                    X,
                    y,
                    indices,
                    absolute_positions[node_id],
                    card_width,
                    card_height,
                    color,
                )
            )
            if np.any(vector):
                vector_lines.append(
                    {
                        "color": color,
                        "text": f"n{node_id}: p_{node_id}={format_sparse_vector(vector)}",
                    }
                )

    return {
        "index": tree_index,
        "title": f"Tree {tree_index + 1}",
        "tile": tile,
        "edges": edges,
        "split_labels": split_labels,
        "cards": cards,
        "vector_lines": vector_lines,
    }


def build_payload(trees, X, y, summary, layout):
    layout_mode = "banner" if layout == "tree4-banner" else "standard"
    tree_indices = list(range(len(trees)))
    if layout == "1x4":
        width, height = 2240, 760
        rows, cols = 1, len(trees)
    elif layout in {"tree4", "tree4-banner"}:
        width, height = (2400, 960) if layout == "tree4-banner" else (1800, 1080)
        rows, cols = 1, 1
        trees = [trees[3]]
        tree_indices = [3]
    else:
        width, height = 1920, 1360
        rows, cols = 2, 2

    margin_x = 56 if layout != "tree4-banner" else 90
    margin_top = 96 if layout != "tree4-banner" else 64
    margin_bottom = 44 if layout != "tree4-banner" else 96
    gap_x = 54
    gap_y = 86
    tile_width = (width - 2 * margin_x - (cols - 1) * gap_x) / cols
    tile_height = (height - margin_top - margin_bottom - (rows - 1) * gap_y) / rows
    payload_trees = []
    for idx, tree in enumerate(trees):
        row = idx // cols
        col = idx % cols
        tile = {
            "x": margin_x + col * (tile_width + gap_x),
            "y": margin_top + row * (tile_height + gap_y),
            "width": tile_width,
            "height": tile_height,
        }
        payload_trees.append(
            build_tree_payload(tree, tree_indices[idx], X, y, tile, layout_mode=layout_mode)
        )

    return {
        "title": "SPORF Methodology: Partitioning High-Dimensional Data with Sparse Projection Trees",
        "layout": layout,
        "text_mode": "axis-only" if layout == "tree4-banner" else "explainer",
        "width": width,
        "height": height,
        "summary": summary,
        "colors": {
            "classes": CLASS_COLORS,
            "background": BG_COLOR,
            "panel_face": PANEL_FACE,
            "panel_edge": PANEL_EDGE,
            "text": TEXT_COLOR,
            "muted": MUTED_TEXT,
            "projection": PROJECTION_COLOR,
            "threshold": THRESHOLD_COLOR,
        },
        "class_legend": [
            {"class_id": class_id, "label": f"class {class_id}", "color": color}
            for class_id, color in enumerate(CLASS_COLORS)
        ],
        "trees": payload_trees,
    }


CSS_TEMPLATE = """\
:root {
  --bg: #090b0e;
  --panel: #080b10;
  --panel-edge: #748090;
  --text: #f3f4f6;
  --muted: #b9c0ca;
  --projection: #ffae43;
  --threshold: #ff4d43;
  --phosphor: #78ff8f;
  --phosphor-soft: rgba(120, 255, 143, 0.32);
  --glow: rgba(120, 190, 255, 0.22);
}

* { box-sizing: border-box; }

body {
  margin: 0;
  min-height: 100vh;
  background:
    radial-gradient(ellipse at 50% 34%, rgba(88, 96, 104, 0.30) 0, rgba(54, 62, 68, 0.18) 34rem, transparent 76rem),
    radial-gradient(ellipse at 24% 72%, rgba(64, 70, 74, 0.16), transparent 44rem),
    radial-gradient(ellipse at 78% 24%, rgba(74, 82, 88, 0.14), transparent 48rem),
    radial-gradient(ellipse at center, transparent 0, transparent 42rem, rgba(0, 0, 0, 0.46) 92rem),
    var(--bg);
  background-color: var(--bg);
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

body::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  background:
    radial-gradient(ellipse at 50% 42%, rgba(210, 222, 226, 0.055), transparent 60rem),
    radial-gradient(ellipse at 50% 8%, rgba(180, 190, 196, 0.040), transparent 48rem);
  mix-blend-mode: screen;
  opacity: 0.80;
}

body::after {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  background-image:
    linear-gradient(rgba(255, 255, 255, 0.018) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255, 255, 255, 0.012) 1px, transparent 1px);
  background-size: 64px 64px, 64px 64px;
  mask-image: radial-gradient(ellipse at 50% 34%, black, transparent 72%);
  opacity: 0.14;
}

.stage {
  width: 100vw;
  margin: 0 auto;
  padding: 0;
}

svg {
  display: block;
  width: 100%;
  height: auto;
  background: transparent;
}

.title {
  fill: var(--text);
  font-size: 30px;
  font-weight: 800;
  letter-spacing: 0.01em;
}

.tree-title {
  fill: var(--text);
  font-size: 20px;
  font-weight: 750;
}

.tree-edge-shadow {
  stroke: rgba(0, 0, 0, 0.62);
  stroke-width: 5.4;
  fill: none;
}

.tree-edge-core {
  stroke: rgba(218, 239, 230, 0.68);
  stroke-width: 2.45;
  fill: none;
  filter: drop-shadow(0 0 3px rgba(186, 230, 214, 0.16));
}

.split-label {
  fill: rgba(238, 247, 244, 0.96);
  font-size: 13px;
  font-weight: 820;
  paint-order: stroke;
  stroke: rgba(1, 3, 6, 0.96);
  stroke-width: 5.5px;
  stroke-linejoin: round;
}

.card {
  fill: rgb(1, 3, 6);
  stroke: rgba(112, 138, 164, 0.90);
  stroke-width: 1.9;
  filter:
    drop-shadow(0 0 8px rgba(132, 255, 70, 0.10))
    drop-shadow(0 0 12px rgba(118, 172, 205, 0.13))
    drop-shadow(0 9px 26px rgba(0, 0, 0, 0.48));
}

.card-bevel-hi {
  fill: none;
  stroke: rgba(224, 248, 255, 0.42);
  stroke-width: 1.65;
}

.card-bevel-lo {
  fill: none;
  stroke: rgba(0, 0, 0, 0.76);
  stroke-width: 1.75;
}

.node-label, .leaf-label {
  font-size: 12px;
  font-weight: 720;
}

.leaf-meta {
  fill: var(--text);
  font-size: 10px;
}

.leaf-n {
  fill: var(--muted);
  font-size: 9px;
}

.tesseract-edge {
  stroke-linecap: round;
  stroke-linejoin: round;
  filter:
    drop-shadow(0 0 2px rgba(120, 255, 143, 0.22))
    drop-shadow(0 0 6px rgba(74, 255, 88, 0.10));
}

.tesseract-edge-glow {
  stroke-linecap: round;
  stroke-linejoin: round;
  filter:
    blur(0.98px)
    drop-shadow(0 0 5.5px rgba(120, 255, 143, 0.29))
    drop-shadow(0 0 11px rgba(74, 255, 88, 0.12));
}

.tesseract-edge.outer { stroke: rgba(155, 255, 176, 0.54); stroke-width: 1.38; }
.tesseract-edge.inner { stroke: rgba(190, 255, 204, 0.66); stroke-width: 1.30; }
.tesseract-edge.cross { stroke: rgba(116, 255, 143, 0.30); stroke-width: 0.92; }
.tesseract-edge-glow.outer { stroke: rgba(120, 255, 143, 0.15); stroke-width: 4.25; }
.tesseract-edge-glow.inner { stroke: rgba(140, 255, 160, 0.18); stroke-width: 3.85; }
.tesseract-edge-glow.cross { stroke: rgba(90, 255, 120, 0.09); stroke-width: 2.85; }

.space-point {
  opacity: 0.58;
  filter: drop-shadow(0 0 2px currentColor);
}

.space-point-halo {
  opacity: 0.12;
  filter: blur(1.0px);
}

.projected-point {
  opacity: 0.96;
  filter:
    drop-shadow(0 0 2px currentColor)
    drop-shadow(0 0 3.5px currentColor);
}

.projected-point-halo {
  opacity: 0.18;
  filter: blur(1.15px);
}

.leaf-dot {
  opacity: 0.92;
  filter: drop-shadow(0 0 2px currentColor);
}

.leaf-dot-halo {
  opacity: 0.15;
  filter: blur(0.9px);
}

.projection-line {
  stroke: var(--projection);
  stroke-width: 1.55;
  opacity: 0.98;
  stroke-linecap: round;
  filter: drop-shadow(0 0 3px rgba(255, 174, 67, 0.35));
}

.projection-line-glow {
  stroke: rgba(255, 174, 67, 0.30);
  stroke-width: 5.0;
  stroke-linecap: round;
  filter: drop-shadow(0 0 7px rgba(255, 174, 67, 0.26));
}

.projection-label {
  fill: var(--projection);
  font-size: 14px;
  font-style: italic;
  font-weight: 700;
  paint-order: stroke;
  stroke: var(--bg);
  stroke-width: 3px;
}

.gutter-line {
  stroke: var(--projection);
  stroke-width: 1.35;
  opacity: 0.86;
  stroke-linecap: round;
  filter: drop-shadow(0 0 3px rgba(255, 174, 67, 0.32));
}

.gutter-line-glow {
  stroke: rgba(255, 174, 67, 0.28);
  stroke-width: 4.6;
  stroke-linecap: round;
  filter: drop-shadow(0 0 7px rgba(255, 174, 67, 0.23));
}

.endpoint-connector {
  stroke: rgba(185, 192, 202, 0.36);
  stroke-width: 1.0;
  stroke-dasharray: 1.4 3.2;
  stroke-linecap: round;
  fill: none;
}

.threshold-band {
  fill: rgba(255, 46, 58, 0.24);
  stroke: rgba(255, 115, 122, 0.58);
  stroke-width: 1.25;
  filter:
    drop-shadow(0 0 4px rgba(255, 46, 58, 0.25))
    drop-shadow(0 0 10px rgba(255, 46, 58, 0.11));
}

.threshold {
  stroke: var(--threshold);
  stroke-width: 3.55;
  stroke-linecap: round;
  filter:
    drop-shadow(0 0 4px rgba(255, 77, 67, 0.58))
    drop-shadow(0 0 9px rgba(255, 77, 67, 0.25));
}

.axis-sign {
  fill: var(--muted);
  font-size: 22px;
  font-weight: 760;
  paint-order: stroke;
  stroke: rgba(1, 3, 6, 0.96);
  stroke-width: 4px;
  stroke-linejoin: round;
}

.side-panel-title {
  fill: var(--text);
  font-size: 14px;
  font-weight: 720;
}

.side-panel-text {
  fill: var(--muted);
  font-size: 12px;
}

.vector-line {
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  font-size: 11px;
}
"""


HTML_TEMPLATE = """\
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SPORF Methodology Visualization</title>
  <link rel="stylesheet" href="__CSS_NAME__">
</head>
<body>
  <main class="stage">
    <svg id="viz" role="img" aria-label="SPORF methodology visualization"></svg>
  </main>
  <script id="payload" type="application/json">
__PAYLOAD_JSON__
  </script>
  <script>
const payload = JSON.parse(document.getElementById("payload").textContent);
const svg = document.getElementById("viz");
const NS = "http://www.w3.org/2000/svg";

function el(name, attrs = {}, parent = svg) {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attrs)) {
    if (value !== undefined && value !== null) node.setAttribute(key, value);
  }
  parent.appendChild(node);
  return node;
}

function colorForClass(classId) {
  return payload.colors.classes[classId % payload.colors.classes.length];
}

function showExplainerText() {
  return payload.text_mode !== "axis-only";
}

function showSplitLabels() {
  return payload.text_mode !== "axis-only" || payload.layout === "tree4-banner";
}

function line(a, b, attrs = {}, parent = svg) {
  return el("line", {x1: a.x, y1: a.y, x2: b.x, y2: b.y, ...attrs}, parent);
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function lerpPoint(a, b, t) {
  return {
    x: lerp(a.x, b.x, t),
    y: lerp(a.y, b.y, t),
  };
}

function intensityFromDepth(depth) {
  // depth=1 is toward the viewer/key light and therefore brighter.
  const z = Math.max(0, Math.min(1, depth));
  const slope = 16.0;
  const midpoint = 0.62;
  const sigmoid = 1 / (1 + Math.exp(-slope * (z - midpoint)));
  const far = 1 / (1 + Math.exp(-slope * (0 - midpoint)));
  const near = 1 / (1 + Math.exp(-slope * (1 - midpoint)));
  const normalized = Math.max(0, Math.min(1, (sigmoid - far) / (near - far)));
  return Math.pow(normalized, 1.55);
}

function ensureDefs() {
  return svg.querySelector("defs") || el("defs");
}

function gradientStopOpacity(pass, intensity) {
  return pass === "glow"
    ? 0.135 + 0.235 * intensity
    : 0.34 + 0.48 * intensity;
}

function gradientColor(kind, pass) {
  if (pass === "glow") {
    if (kind === "inner") return "rgb(140, 255, 160)";
    if (kind === "cross") return "rgb(90, 255, 120)";
    return "rgb(120, 255, 143)";
  }
  if (kind === "inner") return "rgb(190, 255, 204)";
  if (kind === "cross") return "rgb(116, 255, 143)";
  return "rgb(155, 255, 176)";
}

function renderLitLine(a, b, options, parent) {
  const stopCount = options.stops ?? 9;
  for (const pass of ["glow", "core"]) {
    const gradientId = `lit-${pass}-${options.kind}-${options.id}-${Math.random().toString(36).slice(2)}`;
    const gradient = el("linearGradient", {
      id: gradientId,
      gradientUnits: "userSpaceOnUse",
      x1: a.x,
      y1: a.y,
      x2: b.x,
      y2: b.y,
    }, ensureDefs());
    for (let i = 0; i < stopCount; i++) {
      const t = stopCount === 1 ? 0 : i / (stopCount - 1);
      const depth = lerp(options.depthA, options.depthB, t);
      el("stop", {
        offset: `${100 * t}%`,
        "stop-color": gradientColor(options.kind, pass),
        "stop-opacity": gradientStopOpacity(pass, intensityFromDepth(depth)),
      }, gradient);
    }
    line(a, b, {
      class: pass === "glow"
        ? `tesseract-edge-glow ${options.kind}`
        : `tesseract-edge ${options.kind}`,
      style: `stroke: url(#${gradientId});`,
    }, parent);
  }
}

function text(x, y, content, attrs = {}, parent = svg) {
  const node = el("text", {x, y, ...attrs}, parent);
  node.textContent = content;
  return node;
}

function renderCardClip(card, parent) {
  const clipId = `clip-card-${card.type}-${card.id}-${Math.random().toString(36).slice(2)}`;
  const defs = svg.querySelector("defs") || el("defs");
  const clip = el("clipPath", {id: clipId}, defs);
  el("rect", {
    x: card.rect.x,
    y: card.rect.y,
    width: card.rect.width,
    height: card.rect.height,
    rx: 9,
    ry: 9,
  }, clip);
  parent.setAttribute("clip-path", `url(#${clipId})`);
}

function renderCardFrame(card, parent, radius = 9) {
  el("rect", {
    class: "card",
    x: card.rect.x,
    y: card.rect.y,
    width: card.rect.width,
    height: card.rect.height,
    rx: radius,
    ry: radius,
  }, parent);
  el("path", {
    class: "card-bevel-hi",
    d: [
      `M ${card.rect.x + radius} ${card.rect.y + 1.2}`,
      `H ${card.rect.x + card.rect.width - radius}`,
      `M ${card.rect.x + 1.2} ${card.rect.y + radius}`,
      `V ${card.rect.y + card.rect.height - radius}`,
    ].join(" "),
  }, parent);
  el("path", {
    class: "card-bevel-lo",
    d: [
      `M ${card.rect.x + radius} ${card.rect.y + card.rect.height - 1.2}`,
      `H ${card.rect.x + card.rect.width - radius}`,
      `M ${card.rect.x + card.rect.width - 1.2} ${card.rect.y + radius}`,
      `V ${card.rect.y + card.rect.height - radius}`,
    ].join(" "),
  }, parent);
}

function renderInternal(card, parent) {
  renderCardFrame(card, parent, 9);

  const clipped = el("g", {}, parent);
  renderCardClip(card, clipped);
  for (const edge of card.tesseract.edges) {
    renderLitLine(edge.a, edge.b, {
      id: `${card.id}-${edge.id}`,
      kind: edge.kind,
      depthA: edge.depth_a,
      depthB: edge.depth_b,
      stops: 13,
    }, clipped);
  }
  for (const p of card.tesseract.points) {
    el("circle", {
      class: "space-point-halo",
      cx: p.x,
      cy: p.y,
      r: 5.2,
      fill: colorForClass(p.class_id),
    }, clipped);
    el("circle", {
      class: "space-point",
      cx: p.x,
      cy: p.y,
      r: 2.7,
      fill: colorForClass(p.class_id),
    }, clipped);
  }
  if (card.projection) {
    line(card.projection.tesseract_axis.a, card.projection.tesseract_axis.b, {
      class: "projection-line-glow",
    }, clipped);
    line(card.projection.tesseract_axis.a, card.projection.tesseract_axis.b, {
      class: "projection-line",
    }, clipped);
    line(card.projection.gutter_axis.a, card.projection.gutter_axis.b, {
      class: "gutter-line-glow",
    }, clipped);
    line(card.projection.gutter_axis.a, card.projection.gutter_axis.b, {
      class: "gutter-line",
    }, clipped);
    for (const connector of card.projection.connectors) {
      line(connector.a, connector.b, {class: "endpoint-connector"}, clipped);
    }
    if (card.projection.threshold_band) {
      el("polygon", {
        class: "threshold-band",
        points: card.projection.threshold_band.map(p => `${p.x},${p.y}`).join(" "),
      }, clipped);
    }
    line(card.projection.threshold_tick.a, card.projection.threshold_tick.b, {
      class: "threshold",
    }, clipped);
    for (const p of card.projection.projected_points) {
      el("circle", {
        class: "projected-point-halo",
        cx: p.x,
        cy: p.y,
        r: 6.3,
        fill: colorForClass(p.class_id),
      }, clipped);
      el("circle", {
        class: "projected-point",
        cx: p.x,
        cy: p.y,
        r: 3.5,
        fill: colorForClass(p.class_id),
        stroke: payload.colors.background,
        "stroke-width": 0.35,
      }, clipped);
    }
    text(card.projection.negative_label.x, card.projection.negative_label.y, "-", {
      class: "axis-sign",
      "text-anchor": "middle",
      "dominant-baseline": "central",
    }, clipped);
    text(card.projection.positive_label.x, card.projection.positive_label.y, "+", {
      class: "axis-sign",
      "text-anchor": "middle",
      "dominant-baseline": "central",
    }, clipped);
    if (showExplainerText()) {
      text(card.projection.label_position.x, card.projection.label_position.y, card.projection.label, {
        class: "projection-label",
      }, parent);
    }
  }
  if (showExplainerText()) {
    text(card.rect.x + 12, card.rect.y + 18, card.label, {
      class: "node-label",
      fill: card.color,
    }, parent);
  }
}

function renderLeaf(card, parent) {
  renderCardFrame(card, parent, 8);
  if (showExplainerText()) {
    text(card.rect.x + card.rect.width / 2, card.rect.y + 16, card.label, {
      class: "leaf-label",
      fill: card.color,
      "text-anchor": "middle",
    }, parent);
  }
  for (const dot of card.dots) {
    el("circle", {
      class: "leaf-dot-halo",
      cx: dot.x,
      cy: dot.y,
      r: 10.2,
      fill: colorForClass(dot.class_id),
    }, parent);
    el("circle", {
      class: "leaf-dot",
      cx: dot.x,
      cy: dot.y,
      r: 4.7,
      fill: colorForClass(dot.class_id),
    }, parent);
  }
  if (showExplainerText()) {
    text(card.rect.x + card.rect.width / 2, card.rect.y + 33, `class ${card.predicted_class}`, {
      class: "leaf-meta",
      "text-anchor": "middle",
    }, parent);
    text(card.rect.x + card.rect.width / 2, card.rect.y + 43, `n = ${card.n}`, {
      class: "leaf-n",
      "text-anchor": "middle",
    }, parent);
  }
}

function renderSidePanel(tree, parent) {
  if (!showExplainerText()) return;
  if (tree.index !== 0 && payload.layout !== "tree4") return;
  const x = tree.tile.x + tree.tile.width * 0.80;
  let y = tree.tile.y + tree.tile.height * 0.08;
  text(x, y, "How to read a node", {class: "side-panel-title"}, parent);
  y += 22;
  for (const lineText of [
    "orange: projection direction p_i",
    "dotted: p_i endpoint correspondence",
    "on-axis dots: projected values",
    "red tick: threshold value",
  ]) {
    text(x, y, lineText, {class: "side-panel-text"}, parent);
    y += 18;
  }
  y += 24;
  text(x, y, "Classes", {class: "side-panel-title"}, parent);
  y += 22;
  for (const item of payload.class_legend) {
    el("circle", {cx: x + 6, cy: y - 4, r: 5, fill: item.color}, parent);
    text(x + 22, y, item.label, {class: "side-panel-text"}, parent);
    y += 22;
  }
  y += 24;
  text(x, y, "Sparse projection vectors", {class: "side-panel-title"}, parent);
  y += 24;
  for (const item of tree.vector_lines) {
    text(x, y, item.text, {class: "vector-line", fill: item.color}, parent);
    y += 22;
  }
}

function render() {
  svg.setAttribute("viewBox", `0 0 ${payload.width} ${payload.height}`);
  svg.setAttribute("width", payload.width);
  svg.setAttribute("height", payload.height);
  ensureDefs();
  if (showExplainerText()) {
    text(30, 42, payload.title, {class: "title"});
  }
  for (const tree of payload.trees) {
    const group = el("g", {"data-tree": tree.index});
    if (showExplainerText()) {
      text(tree.tile.x, tree.tile.y - 22, tree.title, {class: "tree-title"}, group);
    }
    for (const edge of tree.edges) line(edge.a, edge.b, {class: "tree-edge-shadow"}, group);
    for (const edge of tree.edges) line(edge.a, edge.b, {class: "tree-edge-core"}, group);
    if (showSplitLabels()) {
      for (const label of tree.split_labels) {
        text(label.x, label.y, label.text, {
          class: "split-label",
          "text-anchor": "middle",
          "dominant-baseline": "central",
        }, group);
      }
    }
    for (const card of tree.cards) {
      if (card.type === "leaf") renderLeaf(card, group);
      else renderInternal(card, group);
    }
    renderSidePanel(tree, group);
  }
}

render();
  </script>
</body>
</html>
"""


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(text)
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a tweakable HTML/SVG/CSS methodology visualization from "
            "the saved SPORF methodology forest artifacts."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--layout",
        choices=["2x2", "1x4", "tree4", "tree4-banner"],
        default="2x2",
        help="Web render layout to generate.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=None,
        help="HTML output path. Defaults to <output-dir>/methodology_web_<layout>.html.",
    )
    parser.add_argument(
        "--output-css",
        type=Path,
        default=None,
        help="CSS output path. Defaults to <output-dir>/methodology_web.css.",
    )
    parser.add_argument(
        "--output-payload",
        type=Path,
        default=None,
        help="JSON payload output path. Defaults to <output-dir>/methodology_web_<layout>.json.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir
    X, y = read_points(output_dir / "points.csv")
    trees = read_json(output_dir / "forest_structure.json")
    summary_path = output_dir / "summary.json"
    summary = read_json(summary_path) if summary_path.exists() else None
    payload = build_payload(trees, X, y, summary, args.layout)

    payload_path = args.output_payload or output_dir / f"methodology_web_{args.layout}.json"
    css_path = args.output_css or output_dir / "methodology_web.css"
    html_path = args.output_html or output_dir / f"methodology_web_{args.layout}.html"

    write_json(payload_path, payload)
    write_text(css_path, CSS_TEMPLATE)
    html = (
        HTML_TEMPLATE.replace("__CSS_NAME__", css_path.name)
        .replace("__PAYLOAD_JSON__", json.dumps(payload, indent=2, sort_keys=True))
    )
    write_text(html_path, html)

    print(f"Wrote web payload: {payload_path}")
    print(f"Wrote web CSS: {css_path}")
    print(f"Wrote web HTML: {html_path}")


if __name__ == "__main__":
    main()
