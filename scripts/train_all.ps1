param(
    [switch]$SkipVideo,
    [ValidateSet('bicep','back','chest')]
    [string]$Only
)

# Resolve paths
$Root = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $LogDir "train_all_$Timestamp.log"

# Prefer venv python if present
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$Python = if (Test-Path $VenvPython) { $VenvPython } else { "python" }

# Build command
$cmd = @($Python, "scripts/train_all.py")
if ($SkipVideo) { $cmd += "--skip-video" }
if ($Only) { $cmd += @("--only", $Only) }

Write-Host "[INFO] Logging to $LogPath"
Write-Host "[RUN] $($cmd -join ' ')"

# Execute and tee to log
& $cmd[0] $cmd[1..($cmd.Count - 1)] 2>&1 | Tee-Object -FilePath $LogPath
$exitCode = $LASTEXITCODE

if ($exitCode -ne 0) {
    Write-Host "[ERROR] Training finished with exit code $exitCode" -ForegroundColor Red
    exit $exitCode
}

Write-Host "[DONE] Training complete" -ForegroundColor Green
