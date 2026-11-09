
# InvoiceFlow AI

**AI-assisted invoice data extraction using Python, OpenAI, FastAPI, and LangGraph.**

## Business Problem

Businesses often receive invoices from multiple suppliers in different PDF layouts, including scanned documents. Manually copying supplier details, dates, and amounts into spreadsheets takes time and can introduce errors.

## Solution

InvoiceFlow AI processes uploaded PDF invoices and produces a review-ready CSV file. It extracts eight fields: supplier name, invoice number, invoice date, due date, subtotal, tax, total, and currency. Missing or invalid information is flagged rather than silently accepted.

## Tech Stack

Python · FastAPI · LangGraph · LangChain · OpenAI · Pydantic · PyMuPDF

## Architecture & Workflow

**LangGraph pattern:** Sequential (fixed order using `add_edge()`), **not conditional routing**. The workflow has **4 LangGraph nodes** and **4 custom LangChain tools** (`@tool`). There is **1 LLM-powered extraction step**, but **0 autonomous tool-routing agents**: the model extracts fields; it does not choose which node runs next. The graph is built with `StateGraph()`, `add_node()`, and `add_edge()`, compiled with `graph.compile()`, and executed using `invoice_graph.invoke()`.

```text
PDF Upload (FastAPI)
        |
        v
START -> PDF Reader -> Invoice Extractor -> Invoice Validator -> CSV Exporter -> END
                           (OpenAI LLM)
                                                            |
                                                            v
                                                       CSV Download
```

Only the invoice extractor calls OpenAI. PDF reading, validation, and CSV generation use Python tools. A failed PDF is reported without stopping the other uploaded files.

## Run Locally (Windows CMD + Conda)

```cmd
conda create -n invoiceflow python=3.11 -y
conda activate invoiceflow
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

Start the application:

```cmd
python app.py
```

Open **http://127.0.0.1:8000** in your browser.

## Testing & Current Scope

- Tested with **10 synthetic invoices**, including a scanned PDF.
- Verified that a corrupted PDF can be reported without stopping the remaining batch.
- Current upload limits: **10 PDFs per request**, **10 MB per PDF**, and **5 pages per PDF**.
- Larger batches and real supplier invoices still need testing before production use.

> **Security:** Do not commit your `.env` file or API keys to GitHub.
