@echo off
cd /d "%~dp0.."
set "PYTHON=%WEAVE_PYTHON%"
if not defined PYTHON for /f "delims=" %%P in ('where python 2^>nul') do if not defined PYTHON set "PYTHON=%%P"
if not defined PYTHON (
  echo python not found in PATH; set WEAVE_PYTHON to a python.exe path. 1>&2
  exit /b 1
)
"%PYTHON%" -u scripts\bench_inference_latency.py %*
echo EXITCODE=%ERRORLEVEL%
