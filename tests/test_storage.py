"""SQL flag-rate queries and a small stored-response round trip."""

from __future__ import annotations

import sqlite3

import numpy as np

from forensics.kpi import flag_rate_tables
from forensics.storage import SCHEMA, connect, insert_exam, verify_store, response_checksum
from simulate.design import QUICK
from simulate.generator import simulate


def test_flag_rate_query_matches_hand_counts():
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA)
    conn.executemany(
        """
        INSERT INTO candidates (
            candidate_id, form_id, split, site, week, true_label,
            theta_true, tau_true, total_score
        ) VALUES (?, ?, 'test', ?, ?, 'clean', 0, 0, 0)
        """,
        [
            (1, "A", "S01", 1),
            (2, "A", "S01", 1),
            (3, "A", "S02", 2),
            (4, "B", "S01", 1),
        ],
    )
    conn.executemany(
        "INSERT INTO flags (candidate_id, flagged, combined_score) VALUES (?, ?, ?)",
        [(1, 1, 3.0), (2, 0, 0.2), (3, 1, 2.5), (4, 0, 0.1)],
    )
    conn.commit()
    tables = flag_rate_tables(conn)
    by_form = {row.form: row for row in tables["by_form"].itertuples(index=False)}
    assert int(by_form["A"].n_candidates) == 3
    assert int(by_form["A"].n_flagged) == 2
    assert abs(float(by_form["A"].flag_rate) - (2 / 3)) < 1e-12
    assert int(by_form["B"].n_flagged) == 0
    by_site = {row.site: int(row.n_flagged) for row in tables["by_site"].itertuples(index=False)}
    assert by_site == {"S01": 1, "S02": 1}
    by_week = {int(row.week): int(row.n_flagged) for row in tables["by_week"].itertuples(index=False)}
    assert by_week[1] == 1
    assert by_week[2] == 1
    assert len(tables["by_form_site_week"]) == 3
    conn.close()


def test_sqlite_round_trip_matches_row_count_and_a_stored_candidate():
    exam = simulate(QUICK)
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        conn = connect(Path(tmp) / "exam.sqlite")
        try:
            insert_exam(conn, exam)
            integrity = verify_store(conn, exam, response_checksum(exam))
        finally:
            conn.close()
    assert integrity["response_rows"] == QUICK.n_candidates * QUICK.n_items
    assert integrity["incomplete_candidates"] == 0
    assert len(integrity["spot_checked_candidate_ids"]) == 3
    assert np.isfinite(exam.forms[0].a_true).all()
