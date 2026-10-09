$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path $PSScriptRoot -Parent
$env:PYTHONIOENCODING = 'utf-8'
Set-Location $taskRoot
& (Join-Path $taskRoot '.conda\python.exe') -m sna_project.demo --port 8768
