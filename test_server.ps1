# PowerShell test script for API improvements
Write-Host "`n🚀 Starting API Server..." -ForegroundColor Green
$serverJob = Start-Job -ScriptBlock {
    Set-Location "E:\FYP 1\AI_Gym_FYP"
    & "E:\FYP 1\AI_Gym_FYP\.venv\Scripts\python.exe" api_backend.py
}

Write-Host "⏳ Waiting for server to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

try {
    Write-Host "`n" + ("="*80) -ForegroundColor Cyan
    Write-Host "Testing Root Endpoint (/)" -ForegroundColor Cyan
    Write-Host ("="*80) -ForegroundColor Cyan
    $response = Invoke-RestMethod -Uri "http://localhost:8000/" -Method GET
    $response | ConvertTo-Json
    Write-Host "✅ PASS" -ForegroundColor Green

    Write-Host "`n" + ("="*80) -ForegroundColor Cyan
    Write-Host "Testing Health Endpoint (/health)" -ForegroundColor Cyan
    Write-Host ("="*80) -ForegroundColor Cyan
    $response = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method GET
    $response | ConvertTo-Json
    Write-Host "✅ PASS" -ForegroundColor Green

    Write-Host "`n" + ("="*80) -ForegroundColor Cyan
    Write-Host "Testing Readiness Endpoint (/ready)" -ForegroundColor Cyan
    Write-Host ("="*80) -ForegroundColor Cyan
    $response = Invoke-RestMethod -Uri "http://localhost:8000/ready" -Method GET
    $response | ConvertTo-Json
    Write-Host "✅ PASS" -ForegroundColor Green

    Write-Host "`n" + ("="*80) -ForegroundColor Cyan
    Write-Host "Testing Models Status Endpoint (/api/models/status)" -ForegroundColor Cyan
    Write-Host ("="*80) -ForegroundColor Cyan
    $response = Invoke-RestMethod -Uri "http://localhost:8000/api/models/status" -Method GET
    $response | ConvertTo-Json -Depth 5
    Write-Host "✅ PASS" -ForegroundColor Green

    Write-Host "`n" + ("="*80) -ForegroundColor Green
    Write-Host "🎉 ALL TESTS PASSED!" -ForegroundColor Green
    Write-Host ("="*80) -ForegroundColor Green

} catch {
    Write-Host "`n❌ ERROR: $_" -ForegroundColor Red
} finally {
    Write-Host "`n⏹️ Stopping server..." -ForegroundColor Yellow
    Stop-Job -Job $serverJob
    Remove-Job -Job $serverJob
}
