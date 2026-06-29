$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Sanal ortam bulunamadı. Önce .\setup.ps1 çalıştırın."
}

& .\.venv\Scripts\python.exe -m streamlit run app.py

