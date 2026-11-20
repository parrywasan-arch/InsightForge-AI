import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base project directory (C:\Users\dELL\Downloads\InsightForge-AI)
BASE_DIR = Path(__file__).resolve().parent.parent

# Toggle True when running with valid Azure OpenAI credentials; False for offline test mode
USE_AZURE = os.getenv("USE_AZURE", "False").lower() in ("true", "1", "yes")

# Azure OpenAI Configuration
AZURE_API_KEY = os.getenv("AZURE_API_KEY", "")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-01")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "gpt-4.1")


def _find_csv_file() -> str:
    # Check common locations first
    candidates = [
        BASE_DIR / "sales_data (1).csv",
        BASE_DIR / "sales_data.csv",
        BASE_DIR / "data" / "sales_data (1).csv",
        BASE_DIR / "data" / "sales_data.csv",
    ]
    for path in candidates:
        if path.exists():
            return str(path)

    # Otherwise scan project folders (excluding venv) for any sales CSV
    for root, dirs, files in os.walk(BASE_DIR):
        dirs[:] = [d for d in dirs if d not in ("venv", ".venv", "__pycache__", ".git")]
        for f in files:
            if f.lower().endswith(".csv") and "sales" in f.lower():
                return str(Path(root) / f)

    return str(BASE_DIR / "sales_data.csv")


def _find_pdf_folder() -> str:
    for folder_name in ["PDF Folder", "pdfs", "PDFs", "data/pdfs"]:
        candidate = BASE_DIR / folder_name
        if candidate.exists() and candidate.is_dir():
            return str(candidate)
    return str(BASE_DIR / "PDF Folder")


CSV_PATH = os.getenv("CSV_PATH", _find_csv_file())
DATA_PATH = CSV_PATH
PDF_FOLDER = os.getenv("PDF_FOLDER", _find_pdf_folder())