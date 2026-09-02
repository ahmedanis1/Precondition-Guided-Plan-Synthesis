# Create or update the camara-pgps conda environment (Windows / PowerShell).
#
#   ./scripts/create_env.ps1

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$envName = "camara-pgps"
$exists = (& conda env list) -match "^\s*$envName\s"

if ($exists) {
    Write-Host "Updating existing '$envName' environment..."
    conda env update -f environment.yml --prune
} else {
    Write-Host "Creating '$envName' environment..."
    conda env create -f environment.yml
}

Write-Host ""
Write-Host "Done. Activate with:  conda activate $envName"
