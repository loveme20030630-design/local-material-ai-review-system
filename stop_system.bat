@echo off
setlocal

echo [INFO] Closing project processes on ports 8000 and 5500...

for %%P in (8000 5500) do (
    for /f "tokens=5" %%A in ('netstat -ano ^| findstr /R /C:":%%P .*LISTENING"') do (
        echo [INFO] Closing PID %%A on port %%P
        taskkill /PID %%A /T /F >nul 2>&1
    )
)

echo [INFO] System stopped.
timeout /t 2 /nobreak >nul

endlocal
