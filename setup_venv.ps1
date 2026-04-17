$ErrorActionPreference = "Stop"

$venvPath = ".venv"

# Prefer the Python launcher if available.
$pythonExe = "python"
$pythonArgs = @()
if (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonExe = "py"
    $pythonArgs = @("-3.12")
    & $pythonExe @pythonArgs -c "import sys; print(sys.version_info[:2])" | Out-Null
    if ($LASTEXITCODE -ne 0) {
        $pythonArgs = @()
    }
}

# Enforce Python 3.12+.
& $pythonExe @pythonArgs -c "import sys; assert sys.version_info >= (3,12), 'Python 3.12+ required'"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Python 3.12+ required. Install it and re-run."
    exit 1
}

# Create venv if missing.
if (-not (Test-Path $venvPath)) {
    & $pythonExe @pythonArgs -m venv $venvPath
}

$venvPython = Join-Path $venvPath "Scripts\python.exe"
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt

Write-Host "Done. Activate with: .\\.venv\\Scripts\\Activate.ps1"
