@echo off
setlocal

REM ---------------------------------------------------------------------------
REM Jev Ultrafast on the local Intern-Decision server, driven through Chrome.
REM
REM   start_intern_decision.bat   the decision model on :8011 -- start it first
REM   ChromeDebug.bat             Chrome on --remote-debugging-port=9222
REM   this script                 hands that Chrome to Browser Harness and runs jev
REM
REM The inspector opens at http://127.0.0.1:8766. Browser Harness discovers
REM Chrome's own profiles, not %USERPROFILE%\chromedata, so the browser-level
REM WebSocket URL from the DevTools endpoint is passed through BU_CDP_WS instead.
REM
REM Override any of these before running, e.g.:
REM   set JEV_DIR=D:\other\jev-ultrafast && start_jev.bat
REM ---------------------------------------------------------------------------

if "%JEV_DIR%"==""   set JEV_DIR=%~dp0jev-ultrafast
if "%CHROME%"==""    set CHROME=%ProgramFiles%\Google\Chrome\Application\chrome.exe
if "%PROFILE%"==""   set PROFILE=%USERPROFILE%\chromedata
if "%CDP_PORT%"==""  set CDP_PORT=9222
if "%JEV_PORT%"==""  set JEV_PORT=8766
if "%SYSTEMONE%"=="" set SYSTEMONE=http://127.0.0.1:8011

if not exist "%JEV_DIR%\pyproject.toml" (
    echo [error] no jev-ultrafast checkout at %JEV_DIR%
    echo         set JEV_DIR=^<path^> and try again
    exit /b 1
)

REM ---------------------------------------------------------------------------
REM Stop any jev that is already running, so this copy owns the inspector port
REM and the browser-harness daemon. The daemon holds the Chrome connection, and
REM a second one against the same Chrome is what makes the run flaky.
REM ---------------------------------------------------------------------------
echo   stopping any running jev ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$port = %JEV_PORT%; $self = (Get-CimInstance Win32_Process -Filter ('ProcessId=' + $PID)).ParentProcessId; for ($round = 0; $round -lt 3; $round++) { $pids = @(Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique); $pids += @(Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and ($_.CommandLine -like '*jev.exe*' -or $_.CommandLine -like '*browser_harness.daemon*') } | Select-Object -ExpandProperty ProcessId); $pids += @(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'cmd.exe' -and $_.CommandLine -like '*start_jev.bat*' -and $_.ProcessId -ne $self } | Select-Object -ExpandProperty ProcessId);  $pids = @($pids | Where-Object { $_ } | Sort-Object -Unique); if (-not $pids) { break }; foreach ($p in $pids) { Write-Host ('    stopping pid ' + $p); taskkill.exe /F /T /PID $p 2>&1 | Out-Null }; Start-Sleep -Milliseconds 700 }; for ($i = 0; $i -lt 60; $i++) { if (-not (Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue)) { break }; Start-Sleep -Milliseconds 250 }"

REM The decision model must be up first; Jev has nothing to ask otherwise.
REM The check goes through PowerShell rather than `curl | findstr`. Launched
REM without a console (a shortcut set to hidden, a scheduled task), cmd never
REM hands that pipe an EOF and findstr waits on it forever, so the script
REM stops here with no message.
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r = Invoke-WebRequest -Uri '%SYSTEMONE%/health' -TimeoutSec 5 -UseBasicParsing; exit [int]($r.Content -notlike '*ok*') } catch { exit 1 }"
if errorlevel 1 (
    echo [error] no decision model at %SYSTEMONE%
    echo         run start_intern_decision.bat in another terminal first
    exit /b 1
)

REM Chrome may already be listening on this port; only start it when it is not.
curl.exe -s -m 2 "http://127.0.0.1:%CDP_PORT%/json/version" >nul 2>nul
if errorlevel 1 (
    echo   chrome     : %CHROME%
    echo   profile    : %PROFILE%
    start "" "%CHROME%" --remote-debugging-port=%CDP_PORT% --remote-allow-origins=* --user-data-dir="%PROFILE%"
)

set BU_CDP_WS=
for /f "usebackq delims=" %%u in (`powershell -NoProfile -Command "$d='http://127.0.0.1:%CDP_PORT%/json/version'; for($i=0;$i -lt 60;$i++){ try { (Invoke-RestMethod $d -TimeoutSec 3).webSocketDebuggerUrl; break } catch { Start-Sleep -Milliseconds 500 } }"`) do set BU_CDP_WS=%%u
if not defined BU_CDP_WS (
    echo [error] no remote debugging on port %CDP_PORT%
    echo         run ChromeDebug.bat, then allow remote debugging in Chrome
    exit /b 1
)

echo.
echo   decision   : %SYSTEMONE%
echo   chrome     : %BU_CDP_WS%
echo   inspector  : http://127.0.0.1:%JEV_PORT%
echo.

cd /d "%JEV_DIR%"
set TYPESAFE_DEMO_PORT=%JEV_PORT%
uv run jev
set EXITCODE=%ERRORLEVEL%

if not "%EXITCODE%"=="0" (
    echo.
    echo [error] jev exited with code %EXITCODE%
    pause
)
exit /b %EXITCODE%
