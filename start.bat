@echo off
rem One-click launcher for Windows: checks the setup, starts Cortex, opens the browser.
rem Arguments are passed through, e.g.  start.bat --port 8080
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    rem An existing environment starts immediately and needs no network.
    ".venv\Scripts\python.exe" scripts\demo.py %*
) else (
    where uv >nul 2>nul
    if errorlevel 1 (
        echo uv is not installed. Install it from https://docs.astral.sh/uv/ and run this again.
        pause
        exit /b 1
    )
    echo First run: creating the environment with uv. This downloads the dependencies once.
    uv run python scripts\demo.py %*
)

if errorlevel 1 pause
endlocal
