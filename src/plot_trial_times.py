from __future__ import annotations

import argparse
import math
import re
from pathlib import Path


METHODS = [
    ("YDF RF", "YDF Classical RF", "#1f77b4", "o"),
    ("YDF SPORF", "YDF SPORF", "#d62728", "s"),
    ("cuML", "cuML Classical RF", "#2ca02c", "^"),
    ("SPORF", "cuML SPORF (custom)", "#ff7f0e", "D"),
]

BLOCK_RE = re.compile(
    r"Hyperparameters: .*?'n_train': (?P<n_train>\d+).*?"
    r"YDF RF training time: (?P<ydf_rf_train>[0-9.]+) seconds\s+"
    r"YDF SPORF training time: (?P<ydf_sporf_train>[0-9.]+) seconds\s+"
    r"cuML training time: (?P<cuml_train>[0-9.]+) seconds\s+"
    r"SPORF training time: (?P<sporf_train>[0-9.]+) seconds\s+"
    r"YDF RF prediction time: (?P<ydf_rf_pred>[0-9.]+) seconds\s+"
    r"YDF SPORF prediction time: (?P<ydf_sporf_pred>[0-9.]+) seconds\s+"
    r"cuML prediction time: (?P<cuml_pred>[0-9.]+) seconds\s+"
    r"SPORF prediction time: (?P<sporf_pred>[0-9.]+) seconds",
    flags=re.DOTALL,
)


def parse_results(text: str) -> list[dict[str, float]]:
    rows_by_n_train: dict[int, dict[str, float]] = {}
    for match in BLOCK_RE.finditer(text):
        n_train = int(match.group("n_train"))
        rows_by_n_train[n_train] = {
            "n_train": n_train,
            "YDF RF_train": float(match.group("ydf_rf_train")),
            "YDF SPORF_train": float(match.group("ydf_sporf_train")),
            "cuML_train": float(match.group("cuml_train")),
            "SPORF_train": float(match.group("sporf_train")),
            "YDF RF_pred": float(match.group("ydf_rf_pred")),
            "YDF SPORF_pred": float(match.group("ydf_sporf_pred")),
            "cuML_pred": float(match.group("cuml_pred")),
            "SPORF_pred": float(match.group("sporf_pred")),
        }

    rows = [rows_by_n_train[key] for key in sorted(rows_by_n_train)]
    if not rows:
        raise ValueError("No benchmark blocks found in input log.")
    return rows


def human_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}K"
    return str(value)


def draw_marker(x: float, y: float, marker: str, color: str) -> str:
    if marker == "o":
        return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5.5" fill="{color}" stroke="white" stroke-width="1.5" />'
    if marker == "s":
        return f'<rect x="{x - 5.5:.2f}" y="{y - 5.5:.2f}" width="11" height="11" rx="1.5" fill="{color}" stroke="white" stroke-width="1.5" />'
    if marker == "^":
        points = f"{x:.2f},{y - 6.5:.2f} {x - 6.5:.2f},{y + 5.5:.2f} {x + 6.5:.2f},{y + 5.5:.2f}"
        return f'<polygon points="{points}" fill="{color}" stroke="white" stroke-width="1.5" />'
    if marker == "D":
        points = f"{x:.2f},{y - 6.5:.2f} {x - 6.5:.2f},{y:.2f} {x:.2f},{y + 6.5:.2f} {x + 6.5:.2f},{y:.2f}"
        return f'<polygon points="{points}" fill="{color}" stroke="white" stroke-width="1.5" />'
    return ""


def draw_panel(
    rows: list[dict[str, float]],
    suffix: str,
    title: str,
    ylabel: str,
    x0: float,
    y0: float,
    width: float,
    height: float,
) -> str:
    x_values = [row["n_train"] for row in rows]
    y_values_all = [row[f"{method}_{suffix}"] for row in rows for method, *_ in METHODS]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values_all), max(y_values_all)
    y_min = 10 ** math.floor(math.log10(y_min))
    y_max = 10 ** math.ceil(math.log10(y_max))

    def sx(value: float) -> float:
        return x0 + width * (
            (math.log10(value) - math.log10(x_min)) /
            (math.log10(x_max) - math.log10(x_min))
        )

    def sy(value: float) -> float:
        return y0 + height - height * (
            (math.log10(value) - math.log10(y_min)) /
            (math.log10(y_max) - math.log10(y_min))
        )

    parts = [
        f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" fill="white" rx="10" />',
        f'<text x="{x0 + width / 2:.2f}" y="{y0 - 18:.2f}" text-anchor="middle" font-size="20" font-weight="700" fill="#14213d">{title}</text>',
    ]

    for tick in x_values:
        x = sx(tick)
        parts.append(
            f'<line x1="{x:.2f}" y1="{y0:.2f}" x2="{x:.2f}" y2="{y0 + height:.2f}" stroke="#d9dee7" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{x:.2f}" y="{y0 + height + 28:.2f}" text-anchor="middle" font-size="13" fill="#44556b">{human_count(tick)}</text>'
        )

    power = int(round(math.log10(y_min)))
    while 10 ** power <= y_max * 1.000001:
        value = 10 ** power
        y = sy(value)
        parts.append(
            f'<line x1="{x0:.2f}" y1="{y:.2f}" x2="{x0 + width:.2f}" y2="{y:.2f}" stroke="#d9dee7" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{x0 - 14:.2f}" y="{y + 4:.2f}" text-anchor="end" font-size="13" fill="#44556b">{value:g}</text>'
        )
        power += 1

    parts.extend(
        [
            f'<line x1="{x0:.2f}" y1="{y0 + height:.2f}" x2="{x0 + width:.2f}" y2="{y0 + height:.2f}" stroke="#334155" stroke-width="1.5" />',
            f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x0:.2f}" y2="{y0 + height:.2f}" stroke="#334155" stroke-width="1.5" />',
            f'<text x="{x0 + width / 2:.2f}" y="{y0 + height + 58:.2f}" text-anchor="middle" font-size="15" font-weight="600" fill="#14213d">Training Set Size</text>',
            f'<text x="{x0 - 64:.2f}" y="{y0 + height / 2:.2f}" text-anchor="middle" font-size="15" font-weight="600" fill="#14213d" transform="rotate(-90 {x0 - 64:.2f} {y0 + height / 2:.2f})">{ylabel}</text>',
        ]
    )

    for method, _, color, marker in METHODS:
        points = [(sx(row["n_train"]), sy(row[f"{method}_{suffix}"])) for row in rows]
        path = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round" points="{path}" />'
        )
        for x, y in points:
            parts.append(draw_marker(x, y, marker, color))

    return "\n".join(parts)


def build_svg(rows: list[dict[str, float]]) -> str:
    width = 1440
    height = 760
    panel_width = 520
    panel_height = 400
    left = 120
    top = 180
    gap = 120

    legend_items = []
    legend_x = 250
    legend_y = 670
    for idx, (_, label, color, marker) in enumerate(METHODS):
        x = legend_x + (idx % 2) * 340
        y = legend_y + (idx // 2) * 34
        legend_items.append(f'<line x1="{x:.2f}" y1="{y:.2f}" x2="{x + 28:.2f}" y2="{y:.2f}" stroke="{color}" stroke-width="3.2" stroke-linecap="round" />')
        legend_items.append(draw_marker(x + 14, y, marker, color))
        legend_items.append(f'<text x="{x + 42:.2f}" y="{y + 5:.2f}" font-size="15" fill="#14213d">{label}</text>')

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#f7f4ea" />
<text x="{width / 2:.2f}" y="64" text-anchor="middle" font-size="28" font-weight="700" fill="#14213d">Benchmark Timing vs Training Set Size</text>
<text x="{width / 2:.2f}" y="98" text-anchor="middle" font-size="17" fill="#44556b">YDF classical RF, YDF SPORF, cuML classical RF, and custom cuML SPORF</text>
{draw_panel(rows, "train", "Fit Time Scaling", "Training Time (s)", left, top, panel_width, panel_height)}
{draw_panel(rows, "pred", "Predict Time Scaling", "Prediction Time (s)", left + panel_width + gap, top, panel_width, panel_height)}
<g>
{''.join(legend_items)}
</g>
</svg>
"""


def plot_results(rows: list[dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_svg(rows))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot benchmark training and prediction times from a trial log."
    )
    parser.add_argument(
        "--input",
        default="runs/20260424-1405.txt",
        help="Path to benchmark output log.",
    )
    parser.add_argument(
        "--output",
        default="doc/trial_times_vs_n_train.svg",
        help="Path to output figure.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = parse_results(input_path.read_text())
    plot_results(rows, output_path)


if __name__ == "__main__":
    main()
