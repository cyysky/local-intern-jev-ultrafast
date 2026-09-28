@echo off
setlocal

REM ---------------------------------------------------------------------------
REM Intern-Decision-4B on the transformers backend, served with FastAPI.
REM
REM   POST /v1/systemone        the Jev decision API (vLLM's contract)
REM   POST /v1/chat/completions the same decision from an OpenAI-shaped call
REM   GET  /v1/models           model list
REM   GET  /health              health check
REM
REM Override any of these before running, e.g.:
REM   set DEVICE=cuda:1 && start_intern_decision.bat
REM ---------------------------------------------------------------------------

if "%MODEL_DIR%"=="" set MODEL_DIR=D:\models\Intern-Decision-4B
if "%VENV%"==""      set VENV=D:\vpy314
REM "auto" picks the GPU with the most free memory, so a 10 GB card that is
REM already holding something else does not decide where the model lands.
if "%DEVICE%"==""     set DEVICE=auto
if "%HOST%"==""       set HOST=0.0.0.0
if "%PORT%"==""       set PORT=8011
REM Uncomment to require a bearer token.
REM set API_KEY=change-me

if not exist "%MODEL_DIR%\config.json" (
    echo [error] no checkpoint at %MODEL_DIR%
    echo         set MODEL_DIR=^<path^> and try again
    exit /b 1
)
if not exist "%VENV%\Scripts\python.exe" (
    echo [error] no virtual environment at %VENV%
    echo         set VENV=^<path^> and try again
    exit /b 1
)

REM ---------------------------------------------------------------------------
REM Stop whatever is already running, before this copy loads the checkpoint.
REM
REM Two copies of a 9 GB model do not fit on one 10 GB card. A copy that dies
REM mid-load can also leave a half-grown CUDA segment behind, and the caching
REM allocator then fails every later request with "CUDA out of memory" until the
REM process that owns it is gone. So the port, and any server process that is
REM still alive, goes down first.
REM
REM Deliberately no PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here.
REM On a card this full, a segment that cannot grow fails with cuMemSetAccess
REM and leaves the allocator unable to release blocks, which is the "CUDA out of
REM memory" above. The plain allocator fails cleanly instead, and the server
REM retries once after empty_cache().
REM ---------------------------------------------------------------------------
echo   stopping any running instance ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$port = %PORT%; $self = (Get-CimInstance Win32_Process -Filter ('ProcessId=' + $PID)).ParentProcessId; for ($round = 0; $round -lt 3; $round++) { $pids = @(Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique); $pids += @(Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -like '*intern_decision_server*' } | Select-Object -ExpandProperty ProcessId); $pids += @(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'cmd.exe' -and $_.CommandLine -like '*start_intern_decision.bat*' -and $_.ProcessId -ne $self } | Select-Object -ExpandProperty ProcessId);  $pids = @($pids | Where-Object { $_ } | Sort-Object -Unique); if (-not $pids) { break }; foreach ($p in $pids) { Write-Host ('    stopping pid ' + $p); taskkill.exe /F /T /PID $p 2>&1 | Out-Null }; Start-Sleep -Milliseconds 700 }; for ($i = 0; $i -lt 60; $i++) { if (-not (Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue)) { break }; Start-Sleep -Milliseconds 250 }"

cd /d "%~dp0"
title Intern-Decision systemone on %DEVICE% (port %PORT%)

echo.
echo   checkpoint : %MODEL_DIR%
echo   python     : %VENV%
echo   device     : %DEVICE%
echo   listening  : http://%HOST%:%PORT%
echo   routes     : /v1/systemone  /v1/chat/completions  /v1/models  /health
echo.
echo   In another terminal:
echo.
echo   health and model list:
echo     curl.exe -s http://127.0.0.1:%PORT%/health
echo     curl.exe -s http://127.0.0.1:%PORT%/v1/models
echo.
echo   one field:
echo     curl.exe -s -X POST http://127.0.0.1:%PORT%/v1/systemone -H "content-type: application/json" -d "{\"state\": \"The customer was charged twice and asks for the extra payment back.\", \"questions\": {\"team\": {\"type\": \"choice\", \"instructions\": \"Which team should handle this request?\", \"criteria\": {\"billing\": \"Payments and refunds\", \"delivery\": \"Shipping and delivery\"}}}}"
echo.
echo   every field type at once:
echo     curl.exe -s -X POST http://127.0.0.1:%PORT%/v1/systemone -H "content-type: application/json" -d "{\"state\": \"The customer was charged twice and asks for the extra payment back.\", \"questions\": {\"team\": {\"type\": \"choice\", \"instructions\": \"Which team should handle this request?\", \"criteria\": {\"billing\": \"Payments and refunds\", \"delivery\": \"Shipping and delivery\"}}, \"urgency\": {\"type\": \"score\", \"instructions\": \"Rate the priority.\", \"criteria\": [\"Low\", \"Medium\", \"High\"]}, \"refund_requested\": {\"type\": \"noul\", \"instructions\": \"Is the customer asking for a refund?\"}}}"
echo.
echo   an image as multipart (red lamp):
echo     curl.exe -s -X POST http://127.0.0.1:%PORT%/v1/systemone -F "request={\"state\": \"A traffic signal from a driver viewpoint.\", \"questions\": {\"signal\": {\"type\": \"choice\", \"instructions\": \"Lamp colour?\", \"criteria\": {\"red\": \"red lamp\", \"green\": \"green lamp\", \"yellow\": \"yellow lamp\"}}}}" -F "image=@assets/light_red.png;type=image/png"
echo.
echo   an image as a JSON data URL (see body_image.json):
echo     curl.exe -s -X POST http://127.0.0.1:%PORT%/v1/systemone -H "content-type: application/json" -d "@body_image.json"
echo.
echo   an OpenAI-shaped call (see body_chat.json):
echo     curl.exe -s -X POST http://127.0.0.1:%PORT%/v1/chat/completions -H "content-type: application/json" -d "@body_chat.json"
echo.
echo   an error, a vLLM canvas extension this backend cannot honour:
echo     curl.exe -s -X POST http://127.0.0.1:%PORT%/v1/systemone -H "content-type: application/json" -d "{\"state\": \"s\", \"questions\": {\"t\": {\"type\": \"noul\"}}, \"think\": 4}"
echo.

"%VENV%\Scripts\python.exe" -m uvicorn intern_decision_server:app --host %HOST% --port %PORT% --workers 1
set EXITCODE=%ERRORLEVEL%

if not "%EXITCODE%"=="0" (
    echo.
    echo [error] server exited with code %EXITCODE%
    pause
)
exit /b %EXITCODE%
