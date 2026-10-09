$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path $PSScriptRoot -Parent
$taskEnv = Join-Path $taskRoot '.conda'
$env:CONDA_PKGS_DIRS = Join-Path $taskRoot '.cache\conda'
$env:PIP_CACHE_DIR = Join-Path $taskRoot '.cache\pip'
$taskConda = Join-Path $env:USERPROFILE 'miniconda3\Scripts\conda.exe'
if (-not (Test-Path (Join-Path $taskEnv 'python.exe'))) {
    & $taskConda create --prefix $taskEnv python=3.11 pip -y --use-index-cache --override-channels -c defaults
    if ($LASTEXITCODE -ne 0) { throw 'Conda setup failed' }
}
$taskPython = Join-Path $taskEnv 'python.exe'
& $taskPython -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
& $taskPython -m pip install -r (Join-Path $taskRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& $taskPython -m pip install --no-deps -e $taskRoot
if ($LASTEXITCODE -ne 0) { throw 'Project installation failed' }
Write-Host "Environment ready: $taskEnv"
