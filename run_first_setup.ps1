$ErrorActionPreference = "Stop"
if (-not (Test-Path ".venv")) { python -m venv .venv }
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if (-not (Test-Path ".env")) { Copy-Item .env.example .env; Write-Host "Created .env. Add your CFBD_API_KEY before acquisition." }
python -m jobs.v3_audit_project
Write-Host "Setup complete. Next: python -m jobs.v3_acquire_history --advanced"
