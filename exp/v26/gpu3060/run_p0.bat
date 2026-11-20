@echo off
cd /d I:\Latent_Style\WEAVE
if not exist runs\v26\gpu3060 mkdir runs\v26\gpu3060
set PYTHONUNBUFFERED=1
"C:\Program Files\Python312\python.exe" -u -m exp.v26.gpu3060.queue_p0 > runs\v26\gpu3060\queue.log 2>&1
echo V26_3060_EXIT=%ERRORLEVEL% >> runs\v26\gpu3060\queue.log
