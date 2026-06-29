$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe verify_project.py

Write-Host "Kurulum ve doğrulama tamamlandı." -ForegroundColor Green
Write-Host "Uygulamayı .\run.ps1 ile başlatabilirsiniz."

