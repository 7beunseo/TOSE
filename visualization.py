from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _chart_bounds(values: list[float]) -> tuple[float, float]:
    finite_values = [value for value in values if value == value]
    if not finite_values:
        return 0.0, 1.0
    min_value = min(finite_values)
    max_value = max(finite_values)
    if abs(max_value - min_value) < 1e-8:
        return min_value - 1.0, max_value + 1.0
    margin = 0.1 * (max_value - min_value)
    return min_value - margin, max_value + margin


def _draw_panel(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], title: str) -> None:
    x0, y0, x1, y1 = rect
    draw.rounded_rectangle(rect, radius=14, outline=(180, 180, 180), fill=(252, 252, 252), width=2)
    draw.text((x0 + 12, y0 + 10), title, fill=(40, 40, 40))
    draw.line((x0 + 12, y1 - 28, x1 - 12, y1 - 28), fill=(210, 210, 210), width=1)
    draw.line((x0 + 40, y0 + 36, x0 + 40, y1 - 28), fill=(210, 210, 210), width=1)


def _draw_line_chart(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    title: str,
    values: list[float],
    color: tuple[int, int, int],
) -> None:
    _draw_panel(draw, rect, title)
    x0, y0, x1, y1 = rect
    chart_left = x0 + 42
    chart_right = x1 - 12
    chart_top = y0 + 40
    chart_bottom = y1 - 30
    finite_values = [value for value in values if value == value]
    if len(finite_values) < 1:
        draw.text((chart_left, chart_top + 10), "no data", fill=(120, 120, 120))
        return

    y_min, y_max = _chart_bounds(values)
    draw.text((chart_left, chart_top), f"{values[-1]:.4f}" if values[-1] == values[-1] else "nan", fill=color)
    if len(values) == 1:
        draw.ellipse((chart_left, chart_bottom - 4, chart_left + 8, chart_bottom + 4), fill=color)
        return

    points = []
    for idx, value in enumerate(values):
        if value != value:
            continue
        x = chart_left + (chart_right - chart_left) * idx / max(1, len(values) - 1)
        y = chart_bottom - (chart_bottom - chart_top) * (value - y_min) / max(1e-8, y_max - y_min)
        points.append((x, y))
    if len(points) >= 2:
        draw.line(points, fill=color, width=3)
    for point in points[-5:]:
        draw.ellipse((point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill=color)


def _draw_text_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    title: str,
    lines: list[str],
) -> None:
    _draw_panel(draw, rect, title)
    x0, y0, _, _ = rect
    cursor_y = y0 + 42
    for line in lines:
        draw.text((x0 + 14, cursor_y), line, fill=(55, 55, 55))
        cursor_y += 18


def save_epoch_dashboard(history: list[dict], output_path: Path) -> None:
    """
    Build a compact Pillow dashboard for epoch-level training trends without
    requiring matplotlib.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1500, 980), color=(245, 247, 250))
    draw = ImageDraw.Draw(image)
    _ = ImageFont.load_default()

    total_loss = [float(entry.get("train_loss_total", float("nan"))) for entry in history]
    rmse = [float(entry.get("valid_rmse", float("nan"))) for entry in history]
    f1 = [float(entry.get("valid_bit_f1", float("nan"))) for entry in history]
    lr = [float(entry.get("lr", float("nan"))) for entry in history]
    acc = [float(entry.get("valid_cat_accuracy", float("nan"))) for entry in history]

    panels = [
        (30, 30, 730, 320),
        (770, 30, 1470, 320),
        (30, 350, 730, 640),
        (770, 350, 1470, 640),
        (30, 670, 730, 950),
        (770, 670, 1470, 950),
    ]

    _draw_line_chart(draw, panels[0], "Train Total Loss", total_loss, (27, 94, 32))
    _draw_line_chart(draw, panels[1], "Valid RMSE", rmse, (183, 28, 28))
    _draw_line_chart(draw, panels[2], "Valid Bit F1", f1, (25, 118, 210))
    _draw_line_chart(draw, panels[3], "Learning Rate", lr, (142, 36, 170))
    _draw_line_chart(draw, panels[4], "Valid Cat Accuracy", acc, (0, 121, 107))

    latest = history[-1] if history else {}
    text_lines = [
        f"epoch={latest.get('epoch', 'n/a')}",
        f"global_step={latest.get('global_step', 'n/a')}",
        f"train_forward={latest.get('train_loss_forward', float('nan')):.5f}" if latest else "train_forward=n/a",
        f"train_keep={latest.get('train_loss_forward_keep', float('nan')):.5f}" if latest else "train_keep=n/a",
        f"train_aux={latest.get('train_loss_forward_aux', float('nan')):.5f}" if latest else "train_aux=n/a",
        f"train_num={latest.get('train_loss_forward_num', float('nan')):.5f}" if latest else "train_num=n/a",
        f"train_cat={latest.get('train_loss_forward_cat', float('nan')):.5f}" if latest else "train_cat=n/a",
        f"valid_rmse={latest.get('valid_rmse', float('nan')):.4f}",
        f"valid_bit_f1={latest.get('valid_bit_f1', float('nan')):.4f}",
        f"valid_cat_accuracy={latest.get('valid_cat_accuracy', float('nan')):.4f}",
        f"valid_invalid_code_rate={latest.get('valid_invalid_code_rate', float('nan')):.4f}",
    ]
    _draw_text_panel(draw, panels[5], "Latest Snapshot", text_lines)
    image.save(output_path)
