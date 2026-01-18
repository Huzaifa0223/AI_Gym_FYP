@echo off
echo Starting API Server...
start /B "API Server" "E:\FYP 1\AI_Gym_FYP\.venv\Scripts\python.exe" api_backend.py

echo Waiting for server to start...
timeout /t 10 /nobreak > nul

echo.
echo Testing endpoints...
echo.

curl -s http://localhost:8000/ > test_root.json
echo ✓ Root endpoint
type test_root.json
echo.
echo.

curl -s http://localhost:8000/health > test_health.json
echo ✓ Health endpoint
type test_health.json
echo.
echo.

curl -s http://localhost:8000/ready > test_ready.json
echo ✓ Readiness endpoint
type test_ready.json
echo.
echo.

curl -s http://localhost:8000/api/models/status > test_models.json
echo ✓ Models status endpoint
type test_models.json
echo.
echo.

echo All tests completed!
pause
