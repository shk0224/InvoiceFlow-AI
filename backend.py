"""InvoiceAI: Exactly four LangGraph nodes in sequential order."""
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from tools import (
    read_invoice_pdf, validate_invoice_data, export_invoices_csv,
)
from tools.invoice_extractor import extract_invoice_data

load_dotenv()


class InvoiceState(TypedDict):
    filename: str                  # Current PDF name
    pdf_bytes: bytes               # Uploaded PDF data (not saved to disk)
    pdf_text: str                  # Output of tool 1
    page_images: list[str]         # For scanned invoices
    invoice_data: dict             # Output of tool 2
    validated_invoice: dict        # Output of tool 3
    csv_text: str                  # Output of tool 4
    trace: list[dict]              # UI execution trace


# Node 1: Read a PDF.
def pdf_reader_node(state: InvoiceState) -> dict:
    result = read_invoice_pdf(state["pdf_bytes"])
    return {
        "pdf_text": result["pdf_text"],
        "page_images": result["page_images"],
        "trace": state["trace"] + [{"node": "pdf_reader", "result": f'{result["page_count"]} pages; scanned fallback: {bool(result["page_images"])}'}],
    }


# Node 2: LLM extracts fields.
def invoice_extractor_node(state: InvoiceState) -> dict:
    extracted = extract_invoice_data(state["pdf_text"], state["page_images"])
    return {
        "invoice_data": extracted,
        "trace": state["trace"] + [{"node": "invoice_extractor", "result": "8 fields extracted by OpenAI"}],
    }


# Node 3: Deterministic validation.
def invoice_validator_node(state: InvoiceState) -> dict:
    validated = validate_invoice_data(state["invoice_data"])
    return {
        "validated_invoice": validated,
        "trace": state["trace"] + [{"node": "invoice_validator", "result": validated["status"]}],
    }


# Node 4: Make CSV for this invoice.
def csv_exporter_node(state: InvoiceState) -> dict:
    row = {"filename": state["filename"], **state["validated_invoice"]}
    csv_text = export_invoices_csv([row])
    return {
        "csv_text": csv_text,
        "trace": state["trace"] + [{"node": "csv_exporter", "result": "CSV row generated"}],
    }


# SEQUENTIAL GRAPH (unlike the original conditional Smart Utility graph)
graph = StateGraph(InvoiceState)
graph.add_node("pdf_reader", pdf_reader_node)
graph.add_node("invoice_extractor", invoice_extractor_node)
graph.add_node("invoice_validator", invoice_validator_node)
graph.add_node("csv_exporter", csv_exporter_node)
graph.add_edge(START, "pdf_reader")
graph.add_edge("pdf_reader", "invoice_extractor")
graph.add_edge("invoice_extractor", "invoice_validator")
graph.add_edge("invoice_validator", "csv_exporter")
graph.add_edge("csv_exporter", END)
invoice_graph = graph.compile()


def run_invoice_agent(filename: str, pdf_bytes: bytes) -> dict:
    """Run one invoice through four nodes and return results."""
    initial_state: InvoiceState = {
        "filename": filename, "pdf_bytes": pdf_bytes,
        "pdf_text": "", "page_images": [], "invoice_data": {},
        "validated_invoice": {}, "csv_text": "", "trace": [],
    }
    result = invoice_graph.invoke(initial_state)
    return {"row": {"filename": filename, **result["validated_invoice"]}, "trace": result["trace"]}
