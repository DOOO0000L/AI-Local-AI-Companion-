@echo off
chcp 65001 >nul
title AI智能伴侣 - 一键安装
cd /d "%~dp0"

echo ================================================
echo            AI智能伴侣 - 一键安装
echo ================================================
echo.

REM ---- 1. 检查 Python ----
where python >nul 2>nul
if errorlevel 1 (
    echo [错误] 未检测到 Python!
    echo 请先到 https://www.python.org/downloads/ 安装 Python 3.11 或更高版本,
    echo 安装时务必勾选 "Add python.exe to PATH".
    echo.
    pause
    exit /b 1
)
echo [1/5] 检测到 Python, 创建虚拟环境...
if not exist "venv" (
    python -m venv venv
)
call venv\Scripts\activate.bat

REM ---- 2. 安装基础依赖 ----
echo [2/5] 升级 pip 并安装基础依赖...
python -m pip install --upgrade pip -q
python -m pip install streamlit numpy -q

REM ---- 3. 安装 CUDA 版 llama-cpp-python ----
echo [3/5] 安装 llama-cpp-python (CUDA 加速版)...
python -m pip install "https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.35-cu132/llama_cpp_python-0.3.35-py3-none-win_amd64.whl" -q

REM ---- 4. 下载模型 ----
echo [4/5] 下载模型文件 (约 9GB, 视网速可能需要较长时间, 支持断点续传)...
python download_models.py

REM ---- 5. 检查 CUDA 运行库 ----
echo [5/5] 检查运行环境...
python -c "import llama_cpp; print('  llama-cpp 加载正常')"
if errorlevel 1 (
    echo.
    echo [警告] CUDA 运行库 DLL 缺失!
    echo 启动时若报 "Could not find module ... llama.dll", 请安装 CUDA Toolkit 13.x,
    echo 或查看 README.md 的"常见问题"章节.
)

echo.
echo ================================================
echo     安装完成! 双击 "启动AI伴侣.bat" 开始使用
echo ================================================
pause
