"""SQLite exam store. Responses are inserted in chunks."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np

from simulate.generator import SimulatedExam

SCHEMA = """
CREATE TABLE forms (
    form_id TEXT PRIMARY KEY,
    n_items INTEGER NOT NULL,
    n_leaked INTEGER NOT NULL
);
CREATE TABLE items (
    form_id TEXT NOT NULL,
    item_index INTEGER NOT NULL,
    a_true REAL NOT NULL,
    b_true REAL NOT NULL,
    beta_true REAL NOT NULL,
    leaked INTEGER NOT NULL,
    a_cal REAL,
    b_cal REAL,
    beta_cal REAL,
    PRIMARY KEY (form_id, item_index)
);
CREATE TABLE candidates (
    candidate_id INTEGER PRIMARY KEY,
    form_id TEXT NOT NULL,
    split TEXT NOT NULL,
    site TEXT NOT NULL,
    week INTEGER NOT NULL,
    true_label TEXT NOT NULL,
    theta_true REAL NOT NULL,
    tau_true REAL NOT NULL,
    total_score INTEGER NOT NULL
);
CREATE TABLE responses (
    candidate_id INTEGER NOT NULL,
    item_index INTEGER NOT NULL,
    response INTEGER NOT NULL,
    rt_seconds REAL NOT NULL,
    PRIMARY KEY (candidate_id, item_index)
);
CREATE TABLE flags (
    candidate_id INTEGER PRIMARY KEY,
    flagged INTEGER NOT NULL,
    combined_score REAL NOT NULL
);
"""

FLAG_RATE_BY_FORM_SITE_WEEK = """
SELECT
    c.form_id AS form,
    c.site AS site,
    c.week AS week,
    COUNT(*) AS n_candidates,
    SUM(f.flagged) AS n_flagged,
    AVG(f.flagged) AS flag_rate
FROM candidates AS c
INNER JOIN flags AS f ON f.candidate_id = c.candidate_id
WHERE c.split = 'test'
GROUP BY c.form_id, c.site, c.week
ORDER BY c.form_id, c.site, c.week
"""

FLAG_RATE_BY_FORM = """
SELECT
    c.form_id AS form,
    COUNT(*) AS n_candidates,
    SUM(f.flagged) AS n_flagged,
    AVG(f.flagged) AS flag_rate
FROM candidates AS c
INNER JOIN flags AS f ON f.candidate_id = c.candidate_id
WHERE c.split = 'test'
GROUP BY c.form_id
ORDER BY c.form_id
"""

FLAG_RATE_BY_SITE = """
SELECT
    c.site AS site,
    COUNT(*) AS n_candidates,
    SUM(f.flagged) AS n_flagged,
    AVG(f.flagged) AS flag_rate
FROM candidates AS c
INNER JOIN flags AS f ON f.candidate_id = c.candidate_id
WHERE c.split = 'test'
GROUP BY c.site
ORDER BY c.site
"""

FLAG_RATE_BY_WEEK = """
SELECT
    c.week AS week,
    COUNT(*) AS n_candidates,
    SUM(f.flagged) AS n_flagged,
    AVG(f.flagged) AS flag_rate
FROM candidates AS c
INNER JOIN flags AS f ON f.candidate_id = c.candidate_id
WHERE c.split = 'test'
GROUP BY c.week
ORDER BY c.week
"""

INCOMPLETE_CANDIDATES = """
SELECT c.candidate_id
FROM candidates AS c
INNER JOIN forms AS fo ON fo.form_id = c.form_id
LEFT JOIN responses AS r ON r.candidate_id = c.candidate_id
GROUP BY c.candidate_id, fo.n_items
HAVING COUNT(r.item_index) != fo.n_items
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA cache_size = -32000")
    conn.executescript(SCHEMA)
    return conn


def response_checksum(exam: SimulatedExam) -> str:
    digest = hashlib.sha256()
    for form, split, block in exam.iter_blocks():
        digest.update(form.form_id.encode("utf-8"))
        digest.update(split.encode("utf-8"))
        digest.update(np.ascontiguousarray(block.X).tobytes())
        digest.update(np.ascontiguousarray(block.RT).tobytes())
    return digest.hexdigest()


def expected_response_rows(exam: SimulatedExam) -> int:
    total = 0
    for form, _split, block in exam.iter_blocks():
        total += int(block.X.shape[0] * block.X.shape[1])
    return total


def insert_exam(conn: sqlite3.Connection, exam: SimulatedExam) -> None:
    with conn:
        for form, split, block in exam.iter_blocks():
            if split == "train":
                conn.execute(
                    "INSERT INTO forms (form_id, n_items, n_leaked) VALUES (?, ?, ?)",
                    (form.form_id, int(form.a_true.shape[0]), int(form.leaked_mask.sum())),
                )
                conn.executemany(
                    """
                    INSERT INTO items (
                        form_id, item_index, a_true, b_true, beta_true, leaked
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            form.form_id,
                            i,
                            float(form.a_true[i]),
                            float(form.b_true[i]),
                            float(form.beta_true[i]),
                            int(form.leaked_mask[i]),
                        )
                        for i in range(form.a_true.shape[0])
                    ],
                )
            totals = block.X.sum(axis=1).astype(np.int64)
            conn.executemany(
                """
                INSERT INTO candidates (
                    candidate_id, form_id, split, site, week, true_label,
                    theta_true, tau_true, total_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        int(block.candidate_id[i]),
                        form.form_id,
                        split,
                        str(block.site[i]),
                        int(block.week[i]),
                        str(block.label[i]),
                        float(block.theta[i]),
                        float(block.tau[i]),
                        int(totals[i]),
                    )
                    for i in range(block.X.shape[0])
                ],
            )
            _insert_responses(conn, block.candidate_id, block.X, block.RT)


def _insert_responses(
    conn: sqlite3.Connection,
    candidate_ids: np.ndarray,
    X: np.ndarray,
    RT: np.ndarray,
    chunk_persons: int = 250,
) -> None:
    sql = "INSERT INTO responses (candidate_id, item_index, response, rt_seconds) VALUES (?, ?, ?, ?)"
    n_persons, n_items = X.shape
    items = np.arange(n_items, dtype=np.int32)
    for start in range(0, n_persons, chunk_persons):
        stop = min(start + chunk_persons, n_persons)
        ids = np.repeat(candidate_ids[start:stop], n_items)
        item_col = np.tile(items, stop - start)
        resp = X[start:stop].reshape(-1)
        rt = RT[start:stop].reshape(-1).astype(np.float64)
        conn.executemany(
            sql,
            zip(
                ids.tolist(),
                item_col.tolist(),
                resp.tolist(),
                np.round(rt, 6).tolist(),
            ),
        )


def update_calibration(
    conn: sqlite3.Connection,
    form_id: str,
    a_cal: np.ndarray,
    b_cal: np.ndarray,
    beta_cal: np.ndarray,
) -> None:
    with conn:
        conn.executemany(
            """
            UPDATE items
            SET a_cal = ?, b_cal = ?, beta_cal = ?
            WHERE form_id = ? AND item_index = ?
            """,
            [
                (float(a_cal[i]), float(b_cal[i]), float(beta_cal[i]), form_id, i)
                for i in range(a_cal.shape[0])
            ],
        )


def insert_flags(
    conn: sqlite3.Connection,
    candidate_ids: np.ndarray,
    flagged: np.ndarray,
    combined: np.ndarray,
) -> None:
    with conn:
        conn.executemany(
            "INSERT INTO flags (candidate_id, flagged, combined_score) VALUES (?, ?, ?)",
            [
                (int(candidate_ids[i]), int(flagged[i]), float(combined[i]))
                for i in range(candidate_ids.shape[0])
            ],
        )


def verify_store(conn: sqlite3.Connection, exam: SimulatedExam, checksum: str) -> dict:
    expected = expected_response_rows(exam)
    actual = int(conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0])
    if actual != expected:
        raise RuntimeError(f"Response row count {actual} != {expected}.")
    incomplete = conn.execute(INCOMPLETE_CANDIDATES).fetchall()
    if incomplete:
        raise RuntimeError(f"{len(incomplete)} candidates have an incomplete item record.")
    recomputed = response_checksum(exam)
    if recomputed != checksum:
        raise RuntimeError("Response checksum does not match the manifest.")
    # Spot-check three stored candidates against the in-memory responses.
    sample_ids = []
    for _form, _split, block in exam.iter_blocks():
        sample_ids.append(int(block.candidate_id[0]))
        if len(sample_ids) == 3:
            break
    for cid in sample_ids:
        _assert_candidate_matches(conn, exam, cid)
    return {
        "response_rows": actual,
        "checksum_sha256": checksum,
        "incomplete_candidates": 0,
        "spot_checked_candidate_ids": sample_ids,
    }


def _assert_candidate_matches(conn: sqlite3.Connection, exam: SimulatedExam, candidate_id: int) -> None:
    rows = conn.execute(
        """
        SELECT item_index, response, rt_seconds
        FROM responses
        WHERE candidate_id = ?
        ORDER BY item_index
        """,
        (candidate_id,),
    ).fetchall()
    for form, _split, block in exam.iter_blocks():
        hits = np.where(block.candidate_id == candidate_id)[0]
        if len(hits) == 0:
            continue
        i = int(hits[0])
        got_x = np.array([r[1] for r in rows], dtype=np.uint8)
        got_rt = np.array([r[2] for r in rows], dtype=np.float64)
        if not np.array_equal(got_x, block.X[i]):
            raise RuntimeError(f"Stored responses do not match candidate {candidate_id}.")
        if not np.allclose(got_rt, np.round(block.RT[i].astype(np.float64), 6), atol=1e-5):
            raise RuntimeError(f"Stored response times do not match candidate {candidate_id}.")
        return
    raise RuntimeError(f"Candidate {candidate_id} was not found in the simulation.")
