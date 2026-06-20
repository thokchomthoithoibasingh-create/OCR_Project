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


def _bands_match(row, bands):
    if len(row) != len(bands):
        return False
    return all(abs(_x_left(c) - b) <= COL_TOL for c, b in zip(row, bands))


def detect_blocks(results):
    """Ordered blocks: {'type':'table','rows':[[str,...]]} or {'type':'para','items':[orig OCR items]}."""
    rows = group_rows(results)
    blocks, i, n = [], 0, len(rows)

    while i < n:
        row = rows[i]
        if len(row) >= 2:
            bands = [_x_left(c) for c in row]
            j = i + 1
            multi_count = 1
            while j < n:
                nxt = rows[j]
                if len(nxt) >= 2 and _bands_match(nxt, bands):
                    multi_count += 1
                    j += 1
                elif len(nxt) == 1:
                    if j + 1 < n and len(rows[j + 1]) >= 2 and _bands_match(rows[j + 1], bands):
                        j += 1
                        continue
                    j += 1
                    break
                else:
                    break

            if multi_count >= MIN_TABLE_ROWS:
                table_rows = [[c["text"] for c in r] for r in rows[i:j]]
                blocks.append({"type": "table", "rows": table_rows})
                i = j
                continue

        blocks.append({"type": "para", "items": row})
        i += 1

    return blocks