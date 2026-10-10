"""SQL KPI helpers."""

from __future__ import annotations

import sqlite3

import pandas as pd

from forensics.storage import (
    FLAG_RATE_BY_FORM,
    FLAG_RATE_BY_SITE,
    FLAG_RATE_BY_WEEK,
    FLAG_RATE_BY_FORM_SITE_WEEK,
)


def query_frame(conn: sqlite3.Connection, sql: str) -> pd.DataFrame:
    cursor = conn.execute(sql)
    columns = [item[0] for item in cursor.description]
    return pd.DataFrame(cursor.fetchall(), columns=columns)


def flag_rate_tables(conn: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    return {
        "by_form_site_week": query_frame(conn, FLAG_RATE_BY_FORM_SITE_WEEK),
        "by_form": query_frame(conn, FLAG_RATE_BY_FORM),
        "by_site": query_frame(conn, FLAG_RATE_BY_SITE),
        "by_week": query_frame(conn, FLAG_RATE_BY_WEEK),
    }
