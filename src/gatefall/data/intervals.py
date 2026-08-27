"""Varredura de intervalos sobre segmentos anotados: gaps e sobreposição (agnóstico de dataset)."""


def sweep_gaps_and_overlap(
    segments: list[tuple[float, float]], duration_s: float
) -> tuple[float, float, list[tuple[float, float]]]:
    """Retorna (gap_s, overlap_s, gap_intervals) — varredura por eventos sobre
    segmentos [start, end). Convenção: evento de fim ordena antes de evento de
    início na mesma posição, para que segmentos apenas encostados não fechem
    o gap entre eles nem contem como sobreposição.
    Vídeo sem nenhum segmento: um único gap de duration_s, se duration_s > 0."""
    if not segments:
        if duration_s > 0.0:
            return duration_s, 0.0, [(0.0, duration_s)]
        return 0.0, 0.0, []

    events = sorted(
        [(start, 1) for start, _ in segments] + [(end, -1) for _, end in segments]
    )
    gap_s = 0.0
    overlap_s = 0.0
    gap_intervals: list[tuple[float, float]] = []
    multiplicity = 0
    previous_position = 0.0
    for position, delta in events:
        left = max(previous_position, 0.0)
        right = min(position, duration_s)
        if right > left:
            clipped_length = right - left
            if multiplicity == 0:
                gap_s += clipped_length
                gap_intervals.append((left, right))
            elif multiplicity >= 2:
                overlap_s += multiplicity * (multiplicity - 1) / 2 * clipped_length
        multiplicity += delta
        previous_position = position

    if multiplicity == 0 and previous_position < duration_s:
        gap_s += duration_s - previous_position
        gap_intervals.append((previous_position, duration_s))

    return gap_s, overlap_s, gap_intervals


def tag_gap_positions(
    gap_intervals: list[tuple[float, float]], duration_s: float
) -> list[tuple[float, float, str]]:
    """Classifica cada intervalo de gap (assumidos ordenados por posição, como
    devolvidos por sweep_gaps_and_overlap) como leading (toca o início do
    vídeo), trailing (toca o fim) ou interior (nem um nem outro)."""
    remaining = list(gap_intervals)
    tagged: list[tuple[float, float, str]] = []

    if remaining and remaining[0][0] <= 0.0:
        start, end = remaining[0]
        tagged.append((start, end, "leading"))
        remaining = remaining[1:]

    trailing_item: tuple[float, float] | None = None
    if remaining and remaining[-1][1] >= duration_s:
        trailing_item = remaining[-1]
        remaining = remaining[:-1]

    for start, end in remaining:
        tagged.append((start, end, "interior"))

    if trailing_item is not None:
        tagged.append((trailing_item[0], trailing_item[1], "trailing"))

    return tagged
