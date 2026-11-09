
"""FastAPI entry point. Start with: python app.py"""

from pathlib import Path
import os
import uvicorn

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend import run_invoice_agent
from tools.csv_exporter import export_invoices_csv


app = FastAPI(title="InvoiceFlow AI — LangGraph")

BASE_DIR = Path(__file__).resolve().parent

# Serve external CSS (static/style.css)
app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static"
)

MAX_FILES = 10
MAX_SIZE = 10 * 1024 * 1024


# ==========================================
# FASTAPI ENDPOINT 1: HOME PAGE
# Method: GET /
# ==========================================

@app.get("/")
def home():
    return FileResponse(BASE_DIR / "templates" / "index.html")


# ==========================================
# FASTAPI ENDPOINT 2: HEALTH CHECK
# Method: GET /health
# ==========================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "openai_key_configured": bool(os.getenv("OPENAI_API_KEY"))
    }


# ==========================================
# FASTAPI ENDPOINT 3: PROCESS INVOICES
# Method: POST /api/process
# Calls: backend.py -> run_invoice_agent()
# Also calls: export_invoices_csv.invoke()
# ==========================================

@app.post("/api/process")
async def process_invoices(files: list[UploadFile] = File(...)):
    """Process each invoice separately; one failure must not stop the batch."""

    if not files or len(files) > MAX_FILES:
        raise HTTPException(
            400,
            f"Upload between 1 and {MAX_FILES} PDFs."
        )

    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            503,
            "Set OPENAI_API_KEY in .env before processing invoices."
        )

    rows = []
    traces = []

    for file in files:

        filename = Path(file.filename or "unknown.pdf").name

        try:
            if not filename.lower().endswith(".pdf"):
                raise ValueError("File must end with .pdf")

            content = await file.read(MAX_SIZE + 1)

            if len(content) > MAX_SIZE:
                raise ValueError("File exceeds 10 MB limit")

            result = run_invoice_agent(filename, content)

            rows.append(result["row"])

            traces.append({
                "filename": filename,
                "steps": result["trace"]
            })

        except Exception as error:

            rows.append({
                "filename": filename,
                "status": "Error",
                "issues": str(error)
            })

            traces.append({
                "filename": filename,
                "steps": [
                    {
                        "node": "error",
                        "result": str(error)
                    }
                ]
            })

        finally:
            await file.close()

    return {
        "rows": rows,
        "trace": traces,
        "csv_text": export_invoices_csv.invoke({
            "rows": rows
        })
    }


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
