@echo off
cd /d "%~dp0"

REM 优先使用本地虚拟环境(别人克隆后跑"一键安装.bat"会生成 venv),
REM 否则回退到绝对路径的环境(你自己的电脑)
if exist "venv\Scripts\python.exe" (
    set "PY=venv\Scripts\python.exe"
) else (
    set "PY=D:\PY\PY_TEST\PYproject\.venv\Scripts\python.exe"
)

"%PY%" -m streamlit run "app.py"
pause
