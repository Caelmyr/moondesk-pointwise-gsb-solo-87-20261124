"""Trace one row-wise lunar terminator and classify disk pixels against it.

The scan looks at the horizontal brightness step that runs from the lit side to
the dark side.  A per-row local maximum is only a *candidate*: the terminator is
the candidate chain that stays laterally coherent from row to row.  Rows with no
candidate on the adopted chain stay blank; they are never bridged by a smooth
line, because there is no observation to place an edge there.
"""

import numpy as np

LIGHT, DARK, PENDING = 1, 2, 3

# Path policy.  A long, weak, continuous edge should beat a short, strong spur,
# so strength is capped before it is scored and lateral steps are charged.
TOP_CANDIDATES = 24
LOCAL_RADIUS = 2
SLOPE_TOLERANCE = 2
MAX_GAP = 3
MAX_SLOPE = 4.0
MIN_RUN = 2
CREDIBLE = 2.0
MIN_STEP = 1.0
BLANK_COST = 0.12
START_COST = 0.30
STEP_LINEAR = 0.06
STEP_QUADRATIC = 0.10
ALTERNATIVE_GAP = 3.0
ALTERNATIVE_OFFSET = 4.0
ALTERNATIVE_GAIN = 1.15


def _noise_floor(strength, pairs):
    values = strength[pairs]
    if not values.size:
        return 0.0, 0.0
    median = float(np.median(values))
    deviation = float(np.median(np.abs(values - median)))
    sigma = 1.4826 * deviation
    return sigma, max(MIN_STEP, 1.5 * sigma)


def _candidates(luma, disk, settings):
    """Return per-row local step maxima inside the inspection window."""
    x0, y0, x1, y1 = settings["inspect"]
    cut_luma = luma[y0:y1, x0:x1]
    cut_disk = disk[y0:y1, x0:x1]
    trend = np.diff(cut_luma, axis=1)
    strength = -trend if settings["sun_side"] == "left" else trend
    pairs = cut_disk[:, :-1] & cut_disk[:, 1:]
    sigma, floor = _noise_floor(strength, pairs)

    padded = np.pad(strength, ((0, 0), (1, 1)), constant_values=-np.inf)
    centre = padded[:, 1:-1]
    peak = (centre >= padded[:, :-2]) & (centre >= padded[:, 2:])
    peak &= pairs & (strength >= floor)

    # A lone noisy pixel is not an edge.  Require the step to persist along the
    # edge direction (neighbouring rows within +/-SLOPE_TOLERANCE) rather than
    # with a column neighbour, so real near-vertical edges are kept.
    support = np.zeros_like(peak)
    for shift in range(-SLOPE_TOLERANCE, SLOPE_TOLERANCE + 1):
        if shift == 0:
            source = peak
        elif shift > 0:
            source = np.vstack([np.zeros((shift, peak.shape[1]), dtype=bool), peak[:-shift]])
        else:
            source = np.vstack([peak[-shift:], np.zeros((-shift, peak.shape[1]), dtype=bool)])
        padded_source = np.pad(source, ((0, 0), (1, 1)), constant_values=False)
        for delta in (-1, 0, 1):
            support |= padded_source[:, 1 + delta:peak.shape[1] + 1 + delta]
    peak &= support

    rows, nodes = [], []
    for offset in range(strength.shape[0]):
        indices = np.flatnonzero(peak[offset])
        if indices.size:
            indices = indices[np.argsort(-strength[offset, indices], kind="stable")]
            kept = []
            for index in indices:
                if all(abs(int(index) - other) > LOCAL_RADIUS for other in kept):
                    kept.append(int(index))
                if len(kept) == TOP_CANDIDATES:
                    break
            indices = np.array(sorted(kept))
        rows.append(y0 + offset)
        nodes.append([(x0 + int(index) + 0.5, float(strength[offset, index]))
                      for index in indices])
    return rows, nodes, sigma, floor


def _reward(strength, scale):
    return min(max(strength / scale, 0.0), 1.0)


def _transition_cost(dx, dy):
    slope = abs(dx) / dy
    if slope > MAX_SLOPE:
        return None
    return STEP_LINEAR * slope + STEP_QUADRATIC * slope * slope


def _select(rows, nodes):
    """Chain candidates inside each observed block, never across a real gap.

    A block is a run of rows that each hold at least one candidate.  Blocks are
    separated by rows with no candidate at all; those rows are observations we
    do not have, so no chain is allowed to jump them.  Inside a block a
    transition may skip up to ``MAX_GAP`` candidate-free rows and is charged for
    lateral movement per row, so a long quiet edge beats a strong but
    disconnected spur.
    """
    strengths = [strength for row in nodes for _, strength in row]
    scale = max(2.0, float(np.percentile(strengths, 75))) if strengths else 2.0
    blocks, block = [], []
    for index, row in enumerate(nodes):
        if row:
            block.append(index)
        elif block:
            blocks.append(block)
            block = []
    if block:
        blocks.append(block)

    chosen = {}
    for block in blocks:
        chosen.update(_chain(rows, nodes, block, scale))
    return chosen, scale


def _chain(rows, nodes, block, scale):
    scores = [dict() for _ in block]
    back = [dict() for _ in block]
    for depth, index in enumerate(block):
        for position, (x, strength) in enumerate(nodes[index]):
            reward = _reward(strength, scale)
            best_score, best_prior = reward - START_COST, None
            for skip in range(1, MAX_GAP + 2):
                prior_depth = depth - skip
                if prior_depth < 0:
                    break
                prior_index = block[prior_depth]
                for prior_position, (prior_x, _) in enumerate(nodes[prior_index]):
                    prior = scores[prior_depth].get(prior_position)
                    if prior is None:
                        continue
                    cost = _transition_cost(x - prior_x, index - prior_index)
                    if cost is None:
                        continue
                    candidate = prior + reward - cost - BLANK_COST * (skip - 1)
                    if candidate > best_score:
                        best_score, best_prior = candidate, (prior_depth, prior_position)
            scores[depth][position] = best_score
            back[depth][position] = best_prior

    terminal = max(((depth, position) for depth in range(len(block))
                    for position in scores[depth]),
                   key=lambda state: scores[state[0]][state[1]])
    chosen = {}
    state = terminal
    while state is not None:
        depth, position = state
        chosen[rows[block[depth]]] = nodes[block[depth]][position][0]
        state = back[depth][position]
    return chosen


def _credible(points, nodes_by_row, floor):
    """Reject a chain whose steps sit at the noise floor.

    Wandering through random row maxima can look coherent over a few rows, so
    the adopted steps must clear ``CREDIBLE`` times the measured noise floor.
    Without that margin there is no observed boundary to adopt.
    """
    if not points:
        return False
    strengths = [next(strength for x, strength in nodes_by_row[y] if x == points[y])
                 for y in points]
    return float(np.median(strengths)) >= CREDIBLE * max(floor, MIN_STEP)


def _runs(points, rows):
    runs, run = [], []
    for y in rows:
        if y in points:
            run.append((y, points[y]))
        elif run:
            runs.append(run)
            run = []
    if run:
        runs.append(run)
    return runs


def _repair(points, rows, nodes_by_row):
    """Re-attach a skipped row only when an observed candidate joins both sides.

    The chain search may step over a row whose best point would force an
    over-steep jump.  If that row still has a candidate that sits within the
    slope limit of the adopted rows on *both* sides, it is an observed edge,
    not a gap, so keep it.  Rows without such a candidate stay blank.
    """
    order = {y: index for index, y in enumerate(rows)}
    repaired = dict(points)
    for y in rows:
        if y in repaired:
            continue
        below = next((row for row in rows[order[y] + 1:] if row in repaired), None)
        above = next((row for row in reversed(rows[:order[y]]) if row in repaired), None)
        if above is None or below is None:
            continue
        if below - above > MAX_GAP:
            continue
        for x, _ in nodes_by_row[y]:
            if (abs(x - repaired[above]) <= MAX_SLOPE
                    and abs(x - repaired[below]) <= MAX_SLOPE):
                repaired[y] = x
                break
    return repaired


def _segment(run, nodes_by_row):
    values = [strength for _, x in run
              for candidate, strength in nodes_by_row[_] if candidate == x]
    steps = [abs(run[index][1] - run[index - 1][1]) for index in range(1, len(run))]
    return {
        "y0": run[0][0], "y1": run[-1][0], "rows": len(run),
        "mean_jump": round(float(np.mean(values)), 2) if values else None,
        "max_step": round(float(max(steps)), 2) if steps else 0.0,
        "start_x": round(float(run[0][1]), 2), "end_x": round(float(run[-1][1]), 2),
        "points": [[y, round(float(x), 2)] for y, x in run],
    }


def _alternatives(points, rows, nodes_by_row, segments):
    """Keep only short competitor runs that genuinely rival the adopted edge.

    A candidate must clear the adopted point by more than ``ALTERNATIVE_GAP``,
    be clearly stronger on average than the adopted run in the same rows, and
    be at least ``ALTERNATIVE_OFFSET`` farther from the adopted line on average.
    This drops the many parallel crater-rim steps that sit next to the
    terminator, leaving the one deflection worth comparing.
    """
    adopted_by_row = {}
    for segment in segments:
        for y, x in segment["points"]:
            adopted_by_row[int(y)] = (x, segment["y0"])
    runs, run = [], []
    for y in rows:
        if y not in points:
            continue
        far = [(x, strength) for x, strength in nodes_by_row[y]
               if abs(x - points[y]) > ALTERNATIVE_GAP]
        if not far:
            if run:
                runs.append(run)
            run = []
            continue
        x, strength = max(far, key=lambda candidate: candidate[1])
        if run and y == run[-1][0] + 1 and abs(x - run[-1][1]) <= MAX_SLOPE:
            run.append((y, x, strength))
        else:
            if run:
                runs.append(run)
            run = [(y, x, strength)]
    if run:
        runs.append(run)
    result = []
    for run in runs:
        if len(run) < MIN_RUN:
            continue
        values = [entry[2] for entry in run]
        offsets = [abs(entry[1] - points[entry[0]]) for entry in run]
        adopted_values = [next(strength for x, strength in nodes_by_row[entry[0]]
                               if x == points[entry[0]]) for entry in run]
        mean_offset = float(np.mean(offsets))
        mean_jump = float(np.mean(values))
        adopted_jump = float(np.mean(adopted_values))
        rival = mean_jump >= ALTERNATIVE_GAIN * adopted_jump
        if mean_offset < ALTERNATIVE_OFFSET or not rival:
            continue
        result.append({
            "y0": run[0][0], "y1": run[-1][0], "rows": len(run),
            "mean_jump": round(mean_jump, 2), "adopted_jump": round(adopted_jump, 2),
            "offset": round(mean_offset, 2),
            "segments": sorted({adopted_by_row[entry[0]][1] for entry in run
                                if entry[0] in adopted_by_row}),
            "points": [[y, round(float(x), 2)] for y, x, _ in run],
        })
    return result


def _summary(points, rows, segments, alternatives, primary, floor):
    blank = [y for y in rows if y not in points]
    if not points:
        if floor:
            return (f"窗内没有足够的连续边点（亮度落差未稳定超过噪声底 {round(floor, 2)}），"
                    f"全窗留空；没有用平滑线补接。")
        return "检查窗内没有落在月盘上的行，界点全部留空。"
    text = (f"采用 {len(segments)} 段明暗界，主段连续 {primary['rows']} 行"
            f"（Y {primary['y0']}–{primary['y1']}），平均落差 {primary['mean_jump']}，"
            f"最大行间横移 {primary['max_step']} px。")
    if alternatives:
        alternate = max(alternatives, key=lambda item: (item["rows"], item["mean_jump"]))
        if alternate["rows"] < primary["rows"]:
            text += (f" 另有一段短而强的偏折：连续 {alternate['rows']} 行"
                     f"（Y {alternate['y0']}–{alternate['y1']}），平均落差 {alternate['mean_jump']}，"
                     f"比同段采用边高 {round(alternate['mean_jump'] - alternate['adopted_jump'], 2)}，"
                     f"横向偏离约 {alternate['offset']} px；连续性优先，采用长边，"
                     f"偏折段只作未采用候选。")
        else:
            text += (f" 另有一段 {alternate['rows']} 行的偏折候选（平均落差 {alternate['mean_jump']}），"
                     f"与采用段相当或更弱，未采用。")
    if len(segments) > 1:
        text += f" 其余 {len(segments) - 1} 段也逐行有界点，一并采用。"
    if blank:
        text += f" 另有 {len(blank)} 行缺少可用边点，留空，未作平滑连接。"
    return text


def trace(disk, luma, settings):
    """Adopt at most one edge point per row and describe the choice."""
    rows, candidates, sigma, floor = _candidates(luma, disk, settings)
    points, scale = _select(rows, candidates)
    nodes_by_row = dict(zip(rows, candidates))

    kept = {y: x for y, x in points.items() if x is not None}
    kept = _repair(kept, rows, nodes_by_row)
    runs = [run for run in _runs(kept, rows) if len(run) >= MIN_RUN]
    kept = {y: x for run in runs for y, x in run}
    if not _credible(kept, nodes_by_row, floor):
        kept, runs = {}, []
    segments = [_segment(run, nodes_by_row) for run in runs]
    alternatives = _alternatives(kept, rows, nodes_by_row, segments)
    primary = max(segments, key=lambda segment: (segment["rows"], segment["mean_jump"] or 0),
                  default=None)
    summary = _summary(kept, rows, segments, alternatives, primary, floor)

    flank = _flanks(kept, alternatives, rows)
    table = []
    for y in rows:
        row = {"y": y, "x": None, "jump": None, "candidates": len(nodes_by_row[y]),
               "alternative_x": None, "alternative_jump": None, "note": None}
        if y in kept:
            row["x"] = round(float(kept[y]), 2)
            row["jump"] = round(float(next(strength for x, strength in nodes_by_row[y]
                                           if x == kept[y])), 2)
            far = [(x, strength) for x, strength in nodes_by_row[y]
                   if abs(x - kept[y]) > ALTERNATIVE_GAP]
            if far:
                x, strength = max(far, key=lambda candidate: candidate[1])
                row["alternative_x"] = round(float(x), 2)
                row["alternative_jump"] = round(float(strength), 2)
            row["status"] = "采用"
            row["note"] = flank.get(y)
        elif nodes_by_row[y]:
            row["status"] = "留空"
            row["note"] = "有候选，但与上下界点衔接不上；未采用"
        else:
            row["status"] = "留空"
            row["note"] = "无跨门槛候选"
        table.append(row)

    return {
        "points": [[y, round(float(x), 2)] for y, x in sorted(kept.items())],
        "segments": segments, "alternatives": alternatives, "primary": primary,
        "summary": summary, "rows": table,
        "scale": round(float(scale), 2), "floor": round(float(floor), 2),
        "sigma": round(float(sigma), 2),
    }


def _flanks(points, alternatives, rows):
    """Mark rows where a rejected short deflection was the competing option."""
    candidates = {y: f"未采用短偏折（偏离 {segment['offset']} px）"
                  for segment in alternatives for y, _ in segment["points"]}
    notes = {}
    for y in rows:
        if y in points and y in candidates:
            notes[y] = candidates[y]
    return notes


def window(settings, shape):
    window = np.zeros(shape, dtype=bool)
    x0, y0, x1, y1 = settings["inspect"]
    window[y0:y1, x0:x1] = True
    return window


def labels(disk, settings, points):
    """Classify every disk pixel in the window as light, dark, or pending."""
    height, width = disk.shape
    edge = np.full(height, np.nan)
    for y, x in points:
        if 0 <= y < height:
            edge[int(y)] = float(x)

    columns = np.arange(width)[None, :] + 0.5
    lit_left = settings["sun_side"] == "left"
    left_label, right_label = (LIGHT, DARK) if lit_left else (DARK, LIGHT)
    result = np.where(columns < edge[:, None], left_label, right_label).astype(np.int8)
    inside = disk & window(settings, disk.shape)
    result[~inside] = 0
    result[inside & np.isnan(edge)[:, None]] = PENDING
    return result


def counts(disk, settings, points):
    result = labels(disk, settings, points)
    inside = disk & window(settings, disk.shape)
    totals = {name: int(((result == value) & inside).sum())
              for name, value in (("light", LIGHT), ("dark", DARK), ("pending", PENDING))}
    totals["disk"] = int(inside.sum())
    x0, y0, x1, y1 = settings["inspect"]
    rows = [y for y in range(y0, y1) if disk[y, x0:x1].any()]
    with_edge = {int(y) for y, _ in points}
    totals["boundary_rows"] = sum(1 for y in rows if y in with_edge)
    totals["blank_rows"] = sum(1 for y in rows if y not in with_edge)
    return totals, result
