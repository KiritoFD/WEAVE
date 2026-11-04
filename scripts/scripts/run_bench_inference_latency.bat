@echo off
cd /d I:\Github\Latent_Style\WEAVE
"C:\Program Files\Python312\python.exe" -u scripts\bench_inference_latency.py %*
echo EXITCODE=%ERRORLEVEL%
