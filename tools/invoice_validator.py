
"""Tool 3: Validate extracted fields without inventing missing values."""

from langchain_core.tools import tool
from datetime import date
from decimal import Decimal, InvalidOperation

FIELDS = [
    "supplier_name",
    "invoice_number",
    "invoice_date",
    "due_date",
    "subtotal",
    "tax",
    "total",
    "currency"
]

NUMERIC_FIELDS = ["subtotal", "tax", "total"]


# ==========================================
# TOOL 3: INVOICE VALIDATOR
# Function: validate_invoice_data()
# Called by: backend.py -> invoice_validator_node()
# Purpose: Normalize and validate extracted fields
# LangChain Tool: @tool decorator
# ==========================================

@tool
def validate_invoice_data(invoice: dict) -> dict:
    """Return normalized fields, review status and issue list."""

    clean = {name: invoice.get(name) for name in FIELDS}
    issues = []

    for name in FIELDS:
        if clean[name] is None or clean[name] == "":
            issues.append(f"Missing {name}")
            clean[name] = None

    for name in ["invoice_date", "due_date"]:

        if clean[name] is not None:
            try:
                clean[name] = date.fromisoformat(
                    str(clean[name])
                ).isoformat()

            except (ValueError, TypeError):
                issues.append(
                    f"Invalid {name}: expected YYYY-MM-DD"
                )
                clean[name] = None

    decimals = {}

    for name in NUMERIC_FIELDS:

        if clean[name] is not None:
            try:
                amount = Decimal(str(clean[name]))

                if not amount.is_finite():
                    raise InvalidOperation

                decimals[name] = amount
                clean[name] = format(amount, "f")

            except (InvalidOperation, TypeError, ValueError):
                issues.append(f"Invalid {name}: expected number")
                clean[name] = None

    if clean["currency"] is not None:

        currency = str(clean["currency"]).strip().upper()

        if len(currency) != 3 or not currency.isalpha():
            issues.append("Currency should be a three-letter code")
            clean["currency"] = None
        else:
            clean["currency"] = currency

    if all(name in decimals for name in NUMERIC_FIELDS):

        difference = abs(
            decimals["subtotal"] + decimals["tax"] - decimals["total"]
        )

        if difference > Decimal("0.02"):
            issues.append(
                "Subtotal + tax differs from total (discounts/fees may explain this)"
            )

    return {
        **clean,
        "status": "Review" if issues else "Valid",
        "issues": "; ".join(issues)
    }
