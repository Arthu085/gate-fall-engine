"""Figuras SVG autocontidas (sem dependências de plotagem)."""

from html import escape

PALETTE = ("#1b6ca8", "#d1495b", "#edae49", "#00798c", "#66a182", "#8d6a9f", "#2e4057", "#e07a5f",
           "#3d405b", "#81b29a")
WIDTH = 760


def _svg(height: int, body: list[str]) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
            f'viewBox="0 0 {WIDTH} {height}" font-family="sans-serif" font-size="12">'
            f'<rect width="100%" height="100%" fill="white"/>{"".join(body)}</svg>\n')


def seed_dot_plot(title: str, rows: list[tuple[str, list[float]]], low: float = 0.0, high: float = 1.0) -> str:
    """Uma linha por método: pontos por semente e traço na média."""
    left, right, top, step = 170, WIDTH - 30, 40, 28
    height = top + step * len(rows) + 40
    span = (high - low) or 1.0

    def x(value: float) -> float:
        return left + (right - left) * (min(max(value, low), high) - low) / span

    body = [f'<text x="{left}" y="20" font-weight="bold">{escape(title)}</text>']
    for tick in range(6):
        value = low + span * tick / 5
        body.append(f'<line x1="{x(value):.1f}" y1="{top - 10}" x2="{x(value):.1f}" '
                    f'y2="{height - 30}" stroke="#ddd"/>'
                    f'<text x="{x(value):.1f}" y="{height - 14}" text-anchor="middle">{value:.2f}</text>')
    for index, (label, values) in enumerate(rows):
        y = top + step * index + 8
        body.append(f'<text x="{left - 8}" y="{y + 4}" text-anchor="end">{escape(label)}</text>')
        for value in values:
            body.append(f'<circle cx="{x(value):.1f}" cy="{y}" r="4" fill="{PALETTE[index % len(PALETTE)]}" '
                        f'fill-opacity=".6"/>')
        if values:
            mean = sum(values) / len(values)
            body.append(f'<line x1="{x(mean):.1f}" y1="{y - 9}" x2="{x(mean):.1f}" y2="{y + 9}" '
                        f'stroke="black" stroke-width="2"/>')
    return _svg(height, body)


def scatter(title: str, x_label: str, y_label: str, points: list[tuple[str, float, float]]) -> str:
    left, right, top, bottom = 70, WIDTH - 180, 40, 380
    xs = [point[1] for point in points] or [0.0]; ys = [point[2] for point in points] or [0.0]
    x_high = max(max(xs) * 1.1, 1e-9); y_high = 1.0

    def px(value: float) -> float:
        return left + (right - left) * value / x_high

    def py(value: float) -> float:
        return bottom - (bottom - top) * value / y_high

    body = [f'<text x="{left}" y="20" font-weight="bold">{escape(title)}</text>',
            f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="black"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="black"/>',
            f'<text x="{(left + right) / 2}" y="{bottom + 34}" text-anchor="middle">{escape(x_label)}</text>',
            f'<text x="16" y="{(top + bottom) / 2}" transform="rotate(-90 16 {(top + bottom) / 2})" '
            f'text-anchor="middle">{escape(y_label)}</text>']
    for tick in range(6):
        body.append(f'<text x="{px(x_high * tick / 5):.1f}" y="{bottom + 16}" text-anchor="middle">'
                    f'{x_high * tick / 5:.2f}</text><text x="{left - 6}" y="{py(tick / 5) + 4:.1f}" '
                    f'text-anchor="end">{tick / 5:.1f}</text>')
    labels = sorted({point[0] for point in points})
    for label, x_value, y_value in points:
        color = PALETTE[labels.index(label) % len(PALETTE)]
        body.append(f'<circle cx="{px(x_value):.1f}" cy="{py(y_value):.1f}" r="4" fill="{color}" fill-opacity=".7"/>')
    for index, label in enumerate(labels):
        color = PALETTE[index % len(PALETTE)]
        body.append(f'<circle cx="{right + 20}" cy="{top + 18 * index}" r="5" fill="{color}"/>'
                    f'<text x="{right + 30}" y="{top + 18 * index + 4}">{escape(label)}</text>')
    return _svg(bottom + 50, body)


def timeline(title: str, label_names: tuple[str, ...], truth: list[int], rows: list[tuple[str, list[int], list[int]]],
             fps: float) -> str:
    """Faixa de rótulos verdadeiros e predições por método, com marcas de alarme."""
    left, right, top, band = 150, WIDTH - 20, 40, 22
    count = max(len(truth), 1)
    height = top + band * (len(rows) + 1) + 70 + 16 * ((len(label_names) + 4) // 5)

    def x(index: int) -> float:
        return left + (right - left) * index / count

    def strip(y: int, labels: list[int]) -> list[str]:
        parts = []
        start = 0
        for index in range(1, len(labels) + 1):
            if index == len(labels) or labels[index] != labels[start]:
                value = labels[start]
                color = "#bbbbbb" if value < 0 else PALETTE[value % len(PALETTE)]
                parts.append(f'<rect x="{x(start):.1f}" y="{y}" width="{x(index) - x(start):.2f}" '
                             f'height="{band - 6}" fill="{color}"/>')
                start = index
        return parts

    body = [f'<text x="{left}" y="20" font-weight="bold">{escape(title)}</text>',
            f'<text x="{left - 8}" y="{top + 12}" text-anchor="end">verdade</text>', *strip(top, truth)]
    for index, (label, predicted, alarms) in enumerate(rows, start=1):
        y = top + band * index
        body.append(f'<text x="{left - 8}" y="{y + 12}" text-anchor="end">{escape(label)}</text>')
        body.extend(strip(y, predicted))
        for alarm in alarms:
            body.append(f'<path d="M{x(alarm):.1f} {y - 2} l-4 -6 h8 z" fill="black"/>')
    axis = top + band * (len(rows) + 1) + 6
    seconds = count / fps
    for tick in range(6):
        body.append(f'<text x="{x(round(count * tick / 5)):.1f}" y="{axis + 10}" text-anchor="middle">'
                    f'{seconds * tick / 5:.1f}s</text>')
    for index, name in enumerate(label_names):
        lx = left + 120 * (index % 5); ly = axis + 34 + 16 * (index // 5)
        body.append(f'<rect x="{lx}" y="{ly - 9}" width="10" height="10" fill="{PALETTE[index % len(PALETTE)]}"/>'
                    f'<text x="{lx + 14}" y="{ly}">{escape(name)}</text>')
    return _svg(height, body)
