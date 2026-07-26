@echo off
setlocal

set "PROJECT_DIR=%~dp0"

where conda >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Conda was not found in PATH.
    echo Please open Anaconda Prompt or initialize Conda for Command Prompt.
    pause
    exit /b 1
)

cd /d "%PROJECT_DIR%"
if errorlevel 1 (
    echo [ERROR] Cannot open the project directory:
    echo %PROJECT_DIR%
    pause
    exit /b 1
)

call conda activate llmenv
if errorlevel 1 (
    echo [ERROR] Cannot activate the Conda environment: llmenv
    echo Create the environment and install the required packages first.
    pause
    exit /b 1
)

start "LLM Backend API" cmd /k "cd /d ""%PROJECT_DIR%"" && python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload"

start "LLM Frontend Server" cmd /k "cd /d ""%PROJECT_DIR%web"" && python -m http.server 5500 --bind 127.0.0.1"

timeout /t 3 /nobreak >nul
start "" "http://127.0.0.1:5500/launcher.html"

endlocal
