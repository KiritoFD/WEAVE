@echo off
rem 自定位: %~dp0 = exp\v26\gpu3060\ → 上三级至仓库根 (无绝对路径, 守卫测试合规)
cd /d %~dp0..\..\..
if not exist runs\v26\gpu3060 mkdir runs\v26\gpu3060
set PYTHONUNBUFFERED=1
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u -m exp.v26.gpu3060.queue_p0 > runs\v26\gpu3060\queue.log 2>&1
echo V26_3060_EXIT=%ERRORLEVEL% >> runs\v26\gpu3060\queue.log
