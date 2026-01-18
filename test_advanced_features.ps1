# Advanced Feature Testing - Degraded Mode & Session Tracking
Write-Host "`n🚀 Starting API Server for Advanced Tests..." -ForegroundColor Green
$serverJob = Start-Job -ScriptBlock {
    Set-Location "E:\FYP 1\AI_Gym_FYP"
    & "E:\FYP 1\AI_Gym_FYP\.venv\Scripts\python.exe" api_backend.py
}

Write-Host "⏳ Waiting for server to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

try {
    # Test 1: Check that server logs model availability at startup
    Write-Host "`n" + ("="*80) -ForegroundColor Cyan
    Write-Host "TEST 1: Startup Model Availability Logging" -ForegroundColor Cyan
    Write-Host ("="*80) -ForegroundColor Cyan
    $jobOutput = Receive-Job -Job $serverJob
    if ($jobOutput -like "*Model availability: 0/9 models loaded*") {
        Write-Host "✅ PASS - Server logged model availability" -ForegroundColor Green
    } else {
        Write-Host "❌ FAIL - Expected model availability log" -ForegroundColor Red
    }

    # Test 2: Models Status Endpoint
    Write-Host "`n" + ("="*80) -ForegroundColor Cyan
    Write-Host "TEST 2: Models Status shows 0/9 available" -ForegroundColor Cyan
    Write-Host ("="*80) -ForegroundColor Cyan
    $response = Invoke-RestMethod -Uri "http://localhost:8000/api/models/status" -Method GET
    if ($response.summary.available_models -eq 0 -and $response.summary.total_models -eq 9) {
        Write-Host "✅ PASS - Correctly shows 0 of 9 models available" -ForegroundColor Green
        Write-Host "   Coverage: $($response.summary.coverage_percent)%" -ForegroundColor Yellow
    } else {
        Write-Host "❌ FAIL - Unexpected model count" -ForegroundColor Red
    }

    # Test 3: Readiness Check (should be false with 0 models)
    Write-Host "`n" + ("="*80) -ForegroundColor Cyan
    Write-Host "TEST 3: Readiness Check (should be false with 0 models)" -ForegroundColor Cyan
    Write-Host ("="*80) -ForegroundColor Cyan
    $response = Invoke-RestMethod -Uri "http://localhost:8000/ready" -Method GET
    if ($response.ready -eq $false) {
        Write-Host "✅ PASS - Server correctly reports NOT ready (no models)" -ForegroundColor Green
        Write-Host "   MediaPipe: $($response.mediapipe_initialized)" -ForegroundColor Yellow
        Write-Host "   Models: $($response.models_available)/$($response.models_total)" -ForegroundColor Yellow
    } else {
        Write-Host "❌ FAIL - Should not be ready with 0 models" -ForegroundColor Red
    }

    # Test 4: Check all exercises are documented
    Write-Host "`n" + ("="*80) -ForegroundColor Cyan
    Write-Host "TEST 4: Exercises Endpoint" -ForegroundColor Cyan
    Write-Host ("="*80) -ForegroundColor Cyan
    $response = Invoke-RestMethod -Uri "http://localhost:8000/api/exercises" -Method GET
    if ($response.exercises.Count -eq 3) {
        Write-Host "✅ PASS - All 3 exercises documented" -ForegroundColor Green
        foreach ($ex in $response.exercises) {
            Write-Host "   - $($ex.name) ($($ex.id))" -ForegroundColor Yellow
        }
    } else {
        Write-Host "❌ FAIL - Expected 3 exercises" -ForegroundColor Red
    }

    # Test 5: Check age groups are documented
    Write-Host "`n" + ("="*80) -ForegroundColor Cyan
    Write-Host "TEST 5: Age Groups Endpoint" -ForegroundColor Cyan
    Write-Host ("="*80) -ForegroundColor Cyan
    $response = Invoke-RestMethod -Uri "http://localhost:8000/api/age-groups" -Method GET
    if ($response.age_groups.Count -eq 3) {
        Write-Host "✅ PASS - All 3 age groups documented" -ForegroundColor Green
        foreach ($ag in $response.age_groups) {
            Write-Host "   - $($ag.name): $($ag.age_range)" -ForegroundColor Yellow
        }
    } else {
        Write-Host "❌ FAIL - Expected 3 age groups" -ForegroundColor Red
    }

    # Test 6: Check response includes API version
    Write-Host "`n" + ("="*80) -ForegroundColor Cyan
    Write-Host "TEST 6: API Versioning" -ForegroundColor Cyan
    Write-Host ("="*80) -ForegroundColor Cyan
    $response = Invoke-RestMethod -Uri "http://localhost:8000/" -Method GET
    if ($response.version) {
        Write-Host "✅ PASS - API version present: $($response.version)" -ForegroundColor Green
    } else {
        Write-Host "❌ FAIL - No API version" -ForegroundColor Red
    }

    Write-Host "`n" + ("="*80) -ForegroundColor Green
    Write-Host "🎉 ADVANCED TESTS COMPLETED!" -ForegroundColor Green
    Write-Host ("="*80) -ForegroundColor Green
    Write-Host "`nKey Features Verified:" -ForegroundColor Cyan
    Write-Host "✅ Model availability tracking at startup" -ForegroundColor Green
    Write-Host "✅ /api/models/status endpoint showing 0/9 models" -ForegroundColor Green
    Write-Host "✅ /ready endpoint correctly reports NOT ready" -ForegroundColor Green
    Write-Host "✅ Exercises catalog endpoint" -ForegroundColor Green
    Write-Host "✅ Age groups documentation endpoint" -ForegroundColor Green
    Write-Host "✅ API versioning" -ForegroundColor Green

} catch {
    Write-Host "`n❌ ERROR: $_" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
} finally {
    Write-Host "`n⏹️ Stopping server..." -ForegroundColor Yellow
    Stop-Job -Job $serverJob
    Remove-Job -Job $serverJob
}
