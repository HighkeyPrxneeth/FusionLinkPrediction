param([string]$Device = 'auto')
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path $PSScriptRoot -Parent
$env:PYTHONIOENCODING = 'utf-8'
Push-Location $taskRoot
try {
    & (Join-Path $taskRoot '.conda\python.exe') -m sna_project.run --config configs/main.yaml --device $Device --resume
    if ($LASTEXITCODE -ne 0) { throw 'Experiment failed; inspect artifacts and rerun with resume.' }
    & (Join-Path $taskRoot '.conda\python.exe') -m sna_project.export
    if ($LASTEXITCODE -ne 0) { throw 'Export failed.' }
    & (Join-Path $taskRoot '.conda\python.exe') scripts/audit_demo.py
    if ($LASTEXITCODE -ne 0) { throw 'Demo calculation audit failed.' }
    & (Join-Path $taskRoot '.conda\python.exe') scripts/build_report.py
    if ($LASTEXITCODE -ne 0) { throw 'Report generation failed.' }
    & (Join-Path $taskRoot '.conda\python.exe') scripts/build_submission.py
    if ($LASTEXITCODE -ne 0) { throw 'Submission packaging failed.' }
} finally { Pop-Location }
