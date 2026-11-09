
"""Tool 4: Build CSV output from validated invoice rows."""

import csv
import io
from langchain_core.tools import tool
from .invoice_validator import FIELDS

COLUMNS = ["filename", *FIELDS, "status", "issues"]


# ==========================================
# TOOL 4: CSV EXPORTER
# Function: export_invoices_csv()
# Called by: backend.py and app.py
# Purpose: Convert validated invoices to CSV
# LangChain Tool: @tool decorator
# ==========================================

@tool
def export_invoices_csv(rows: list[dict]) -> str:
    """Generate one CSV row per invoice; escape spreadsheet formulas."""

    output = io.StringIO(newline="")

    writer = csv.DictWriter(
        output,
        fieldnames=COLUMNS,
        extrasaction="ignore"
    )

    writer.writeheader()

    for row in rows:
        safe = {}

        for field in COLUMNS:
            value = row.get(field)
            value = "" if value is None else str(value)

            # Prevent formula injection when the CSV is opened in spreadsheet software.
            if value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")) and field not in ["subtotal", "tax", "total"]:
                value = "'" + value

            safe[field] = value

        writer.writerow(safe)

    return output.getvalue()
