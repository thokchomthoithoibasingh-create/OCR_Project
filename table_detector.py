"""Detects table regions in OCR results using row/column position clustering."""

ROW_TOL = 15
COL_TOL = 25
MIN_TABLE_ROWS = 3


def _y_center(item):
    ys = [p[1] for p in item["box"]]
    return sum(ys) / len(ys)


def _x_left(item):
    return min(p[0] for p in item["box"])


def group_rows(results):
    if not results:
        return []
    rows, cur, cur_y = [], [results[0]], _y_center(results[0])
    for item in results[1:]:
        y = _y_center(item)
        if abs(y - cur_y) <= ROW_TOL:
            cur.append(item)
        else:
            rows.append(sorted(cur, key=_x_left))
            cur, cur_y = [item], y
    rows.append(sorted(cur, key=_x_left))
    return rows


def _align_table_rows(row_groups):
    """
    Snap every row's cells into a consistent set of K columns, using the
    average x-position of rows that detected the 'full' cell count as the
    reference column bands. Rows with fewer detected cells (OCR missed a
    box) get their cells placed into the nearest matching column instead
    of being shoved left into columns 1,2,3...

    A single-cell "stray" row (e.g. one line of a wrapped cell, like a
    long school name that spills onto the line above/below its row) is
    merged into the neighboring row's matching empty column instead of
    being emitted as its own line.
    """
    multi_lens = [len(r) for r in row_groups if len(r) >= 2]
    if not multi_lens:
        return [[c["text"] for c in r] for r in row_groups]

    k = max(set(multi_lens), key=multi_lens.count)
    ref_rows = [r for r in row_groups if len(r) == k]
    if not ref_rows:
        k = max(multi_lens)
        ref_rows = [r for r in row_groups if len(r) == k]

    bands = [sum(_x_left(r[col]) for r in ref_rows) / len(ref_rows) for col in range(k)]

    n = len(row_groups)
    aligned = [None] * n
    original_filled = [set() for _ in range(n)]

    for idx, r in enumerate(row_groups):
        if len(r) >= 2:
            row_out = [""] * k
            for c in r:
                nearest = min(range(k), key=lambda i: abs(bands[i] - _x_left(c)))
                row_out[nearest] = c["text"]
                original_filled[idx].add(nearest)
            aligned[idx] = row_out

    for idx, r in enumerate(row_groups):
        if len(r) != 1:
            continue
        text = r[0]["text"]
        col = min(range(k), key=lambda i: abs(bands[i] - _x_left(r[0])))
        target = None
        if idx - 1 >= 0 and aligned[idx - 1] is not None and col not in original_filled[idx - 1]:
            target = idx - 1
        elif idx + 1 < n and aligned[idx + 1] is not None and col not in original_filled[idx + 1]:
            target = idx + 1
        if target is not None:
            cur = aligned[target][col]
            aligned[target][col] = f"{cur} {text}".strip() if cur else text
        else:
            row_out = [""] * k
            row_out[0] = text
            aligned[idx] = row_out

    return [row for row in aligned if row is not None]


def detect_blocks(results):
    """
    Ordered blocks: {'type':'table','rows':[[str,...]]} or {'type':'para','items':[orig OCR items]}.

    Any consecutive run of 3+ rows with 2+ cells is treated as one table
    (single-cell rows in between, e.g. section headers/sub-totals, are
    kept inside the run as spanning rows). Column x-position is NOT
    required to match exactly across rows - real scanned OCR has too
    much pixel jitter for that to be reliable, and it was causing the
    table to split mid-way and fall back into jumbled paragraph text.
    """
    rows = group_rows(results)
    blocks, i, n = [], 0, len(rows)

    while i < n:
        if len(rows[i]) >= 2:
            j = i
            multi_count = 0
            while j < n:
                if len(rows[j]) >= 2:
                    multi_count += 1
                    j += 1
                elif len(rows[j]) == 1 and j + 1 < n and len(rows[j + 1]) >= 2:
                    j += 1  # lone row sandwiched between table rows -> spanning row
                else:
                    break

            if multi_count >= MIN_TABLE_ROWS:
                table_rows = _align_table_rows(rows[i:j])
                blocks.append({"type": "table", "rows": table_rows})
                i = j
                continue

        blocks.append({"type": "para", "items": rows[i]})
        i += 1

    return blocks