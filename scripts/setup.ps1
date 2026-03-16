$ErrorActionPreference = "Stop"

Write-Host "== Creating venv =="
python -m venv .venv

Write-Host "== Activating venv =="
. .\.venv\Scripts\Activate.ps1

Write-Host "== Upgrading pip =="
python -m pip install --upgrade pip

Write-Host "== Installing dependencies =="
pip install -r requirements.txt

Write-Host "== Setup complete =="
