<#
Launcher that keeps this project's disk activity on the D drive.

Python defaults to writing temp files and its pip cache under the user profile on
C:, which is the small drive on this machine. This points both at D: before
handing off to the virtual environment.

Usage:
    .\run.ps1            # start the app
    .\run.ps1 test       # run both test suites
    .\run.ps1 install    # install or update dependencies
#>
param(
    [ValidateSet("app", "test", "install")]
    [string]$Task = "app",

    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
$project = $PSScriptRoot

# Keep scratch space and the package cache off C:.
$tempDir = "D:\pytmp"
$cacheDir = "D:\pycache"
New-Item -ItemType Directory -Force -Path $tempDir, "$cacheDir\pip" | Out-Null

$env:TEMP = $tempDir
$env:TMP = $tempDir
$env:PIP_CACHE_DIR = "$cacheDir\pip"
$env:XDG_CACHE_HOME = $cacheDir
# Keep __pycache__ trees out of the source folder and on D as well.
$env:PYTHONPYCACHEPREFIX = "$cacheDir\pycache"

$python = Join-Path $project "venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "No virtual environment found. Create one with: python -m venv venv"
}

Push-Location $project
try {
    switch ($Task) {
        "install" {
            & $python -m pip install -r requirements.txt
        }
        "test" {
            & $python test_inventory.py
            & $python test_retrieval.py
        }
        "app" {
            & $python -m streamlit run app.py --server.port $Port
        }
    }
}
finally {
    Pop-Location
}
