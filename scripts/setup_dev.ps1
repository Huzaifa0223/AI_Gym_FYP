<#
.SYNOPSIS
    Idempotent dev-environment bootstrap for AI Gym Trainer FYP (Windows / PowerShell).

.DESCRIPTION
    1. Verifies Python 3.11 is on PATH.
    2. Creates .venv if absent; activates it.
    3. Upgrades pip; installs pinned requirements.txt.
    4. Verifies Git for Windows is installed (required by Claude Code).
    5. Installs Claude Code native binary if not already on PATH.
    6. Prints a success banner with next-step commands.

.NOTES
    Run from the repo root:
        powershell -ExecutionPolicy Bypass -File scripts\setup_dev.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ── Helpers ──────────────────────────────────────────────────────────────────
function Write-Step {
    param([string]$Message)
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Write-OK {
    param([string]$Message)
    Write-Host "    [OK] $Message" -ForegroundColor Green
}

function Write-Fail {
    param([string]$Message, [int]$ExitCode = 1)
    Write-Host "`n[FAIL] $Message" -ForegroundColor Red
    exit $ExitCode
}

# ── Resolve repo root (script lives in scripts/) ─────────────────────────────
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
Write-Host "Repo root: $RepoRoot" -ForegroundColor DarkGray

# ── Step 1: Verify Python 3.11 ───────────────────────────────────────────────
Write-Step "Checking Python version"

$PythonCmd = $null
foreach ($candidate in @('python', 'python3', 'python3.11')) {
    try {
        $ver = & $candidate --version 2>&1
        if ($ver -match '3\.11\.\d+') {
            $PythonCmd = $candidate
            break
        }
    } catch { continue }
}

if ($null -eq $PythonCmd) {
    Write-Fail @"
Python 3.11 was not found on PATH.

Install it from https://www.python.org/downloads/release/python-3110/
(tick "Add Python to PATH" during setup), then re-run this script.
"@
}

$PythonVersion = & $PythonCmd --version
Write-OK "Found: $PythonVersion  (command: $PythonCmd)"

# ── Step 2: Create / activate .venv ──────────────────────────────────────────
Write-Step "Preparing virtual environment"

$VenvDir = Join-Path $RepoRoot '.venv'
if (-not (Test-Path $VenvDir)) {
    Write-Host "    Creating .venv ..."
    & $PythonCmd -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { Write-Fail "Failed to create virtual environment." }
    Write-OK ".venv created at $VenvDir"
} else {
    Write-OK ".venv already exists — skipping creation"
}

$PipCmd     = Join-Path $VenvDir 'Scripts\pip.exe'
$PythonVenv = Join-Path $VenvDir 'Scripts\python.exe'

if (-not (Test-Path $PipCmd)) {
    Write-Fail ".venv pip not found at $PipCmd — try deleting .venv and re-running."
}

# ── Step 3: Upgrade pip + install requirements ────────────────────────────────
Write-Step "Installing Python dependencies"

Write-Host "    Upgrading pip ..."
& $PipCmd install --quiet --upgrade pip
if ($LASTEXITCODE -ne 0) { Write-Fail "pip upgrade failed." }

$ReqFile = Join-Path $RepoRoot 'requirements.txt'
if (-not (Test-Path $ReqFile)) {
    Write-Fail "requirements.txt not found at $ReqFile"
}

Write-Host "    Installing requirements.txt ..."
& $PipCmd install --quiet -r $ReqFile
if ($LASTEXITCODE -ne 0) { Write-Fail "pip install -r requirements.txt failed." }

Write-OK "All Python dependencies installed"

# ── Step 4: Git for Windows ───────────────────────────────────────────────────
Write-Step "Checking Git for Windows"

$GitCmd = Get-Command git.exe -ErrorAction SilentlyContinue
if ($null -eq $GitCmd) {
    Write-Fail @"
Git for Windows was not found on PATH.

Claude Code (native Windows) requires Git for Windows to execute shell commands.
Download and install it from: https://git-scm.com/download/win

After installation, restart PowerShell and re-run this script.
"@ -ExitCode 2
}

$GitVersion = & git.exe --version 2>&1
Write-OK "Found: $GitVersion  (path: $($GitCmd.Source))"

# ── Step 5: Claude Code native binary ────────────────────────────────────────
Write-Step "Checking Claude Code installation"

$script:ClaudeJustInstalled = $false

$ClaudeOnPath = Get-Command claude -ErrorAction SilentlyContinue
if ($ClaudeOnPath) {
    try {
        $claudeVer = & claude --version 2>&1
        Write-OK "Claude Code already installed: $claudeVer ($($ClaudeOnPath.Source))"
    } catch {
        Write-Host "    [WARN] claude found on PATH but version check failed: $_" -ForegroundColor Yellow
    }
} else {
    Write-Host "    Claude Code not found — running official installer ..." -ForegroundColor Yellow
    try {
        Invoke-Expression (Invoke-RestMethod https://claude.ai/install.ps1)
    } catch {
        Write-Fail "Claude Code installer failed: $_" -ExitCode 3
    }

    $ClaudeExePath = Join-Path $env:USERPROFILE '.local\bin\claude.exe'
    if (Test-Path $ClaudeExePath) {
        $script:ClaudeJustInstalled = $true
        $ClaudeNowOnPath = Get-Command claude -ErrorAction SilentlyContinue
        if ($ClaudeNowOnPath) {
            try {
                $claudeVer = & claude --version 2>&1
                Write-OK "Claude Code installed and active: $claudeVer"
            } catch {
                Write-Host "    [WARN] Installed but version check failed: $_" -ForegroundColor Yellow
            }
        } else {
            Write-Host ""
            Write-Host "    [WARN] Claude Code was installed but is not yet resolvable in this" -ForegroundColor Yellow
            Write-Host "           PowerShell session. PATH is only updated in NEW sessions."   -ForegroundColor Yellow
            Write-Host ""
            Write-Host "    Action required:"                          -ForegroundColor White
            Write-Host "      1. Close this terminal."                 -ForegroundColor Yellow
            Write-Host "      2. Open a new PowerShell window."        -ForegroundColor Yellow
            Write-Host "      3. Run: claude --version"                -ForegroundColor Yellow
            Write-Host ""
        }
    } else {
        Write-Fail "Claude Code installer ran but binary not found at: $ClaudeExePath" -ExitCode 3
    }
}

# ── Step 6: Success banner ─────────────────────────────────────────────────────
Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║          AI Gym Trainer FYP — Dev Environment Ready         ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "  Activate venv (if not already active):" -ForegroundColor White
Write-Host "    .\.venv\Scripts\Activate.ps1"         -ForegroundColor Yellow
Write-Host ""
Write-Host "  Start the CV loop:"                     -ForegroundColor White
Write-Host "    python main.py"                        -ForegroundColor Yellow
Write-Host ""
Write-Host "  Start the REST API:"                    -ForegroundColor White
Write-Host "    uvicorn api_backend:app --reload --host 0.0.0.0 --port 8000" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Run tests:"                             -ForegroundColor White
Write-Host "    pytest tests/ -v"                     -ForegroundColor Yellow
Write-Host ""
Write-Host "  Train all 9 ML models:"                 -ForegroundColor White
Write-Host "    python training/train_universal.py"   -ForegroundColor Yellow
Write-Host ""
if ($script:ClaudeJustInstalled) {
    Write-Host "  NOTE: Claude Code was just installed. Open a NEW terminal before" -ForegroundColor Yellow
    Write-Host "        running 'claude' — PATH changes take effect in new sessions." -ForegroundColor Yellow
    Write-Host ""
}
