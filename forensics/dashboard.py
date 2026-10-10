"""Static PNG and HTML dashboard for evaluation-cohort flag rates."""

from __future__ import annotations

import base64
import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _bar(ax, labels, values, title: str, ylabel: str) -> None:
    ax.bar(labels, values, color="#1f4e79")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(0.05, float(max(values)) * 1.25 if len(values) else 0.05))
    ax.tick_params(axis="x", labelrotation=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_flag_dashboard(tables: dict[str, pd.DataFrame], png_path: Path, html_path: Path) -> None:
    by_form = tables["by_form"]
    by_site = tables["by_site"]
    by_week = tables["by_week"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    _bar(axes[0], by_form["form"].astype(str), by_form["flag_rate"].astype(float), "Flag rate by form", "Flag rate")
    _bar(axes[1], by_site["site"].astype(str), by_site["flag_rate"].astype(float), "Flag rate by site", "Flag rate")
    _bar(axes[2], by_week["week"].astype(str), by_week["flag_rate"].astype(float), "Flag rate by week", "Flag rate")
    fig.suptitle("Simulated evaluation cohort: combined-score flags")
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    _write_html(tables, png_path, html_path)


def _table_html(frame: pd.DataFrame) -> str:
    headers = "".join(f"<th>{html.escape(str(col))}</th>" for col in frame.columns)
    body = []
    for row in frame.itertuples(index=False):
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in row)
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _write_html(tables: dict[str, pd.DataFrame], png_path: Path, html_path: Path) -> None:
    encoded = base64.b64encode(png_path.read_bytes()).decode("ascii")
    cell = tables["by_form_site_week"].copy()
    for col in cell.columns:
        if col == "flag_rate":
            cell[col] = cell[col].map(lambda v: f"{float(v):.4f}")
    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Test-security flag dashboard</title>
<style>
body {{ font-family: Georgia, serif; margin: 2rem; color: #1c1c1c; }}
table {{ border-collapse: collapse; margin: 1rem 0 2rem; }}
th, td {{ border: 1px solid #ccc; padding: 0.35rem 0.6rem; text-align: right; }}
th {{ background: #f4f7fb; }}
img {{ max-width: 100%; height: auto; }}
</style>
</head>
<body>
<h1>Simulated exam flag rates</h1>
<p>Flags are evaluation-cohort candidates whose combined score is above the threshold set for a 1% false-positive rate on the clean validation slice. Counts come from SQLite queries grouped by form, site, and week. These are simulated candidates, not an operational testing program.</p>
<img alt="Flag rates by form, site, and week" src="data:image/png;base64,{encoded}"/>
<h2>By form</h2>
{_table_html(tables["by_form"])}
<h2>By site</h2>
{_table_html(tables["by_site"])}
<h2>By week</h2>
{_table_html(tables["by_week"])}
<h2>By form, site, and week</h2>
{_table_html(cell)}
</body>
</html>
"""
    html_path.write_text(document, encoding="utf-8")
