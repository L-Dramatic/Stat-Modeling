@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

echo.
echo ╔════════════════════════════════════════════════╗
echo ║     城市空气质量监测预警系统 v2.0              ║
echo ║     AirQuality-StatModel-2025                  ║
echo ╚════════════════════════════════════════════════╝
echo.

:: 切换到脚本所在目录
cd /d "%~dp0"

:: ========== 检查 Python 环境 ==========
echo [1/3] 检查 Python 环境...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到 Python，请先安装 Python 3.8+
    echo        下载地址: https://www.python.org/downloads/
    goto :error
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo        Python 版本: %PYTHON_VERSION%

:: ========== 检查虚拟环境 ==========
echo [2/3] 检查虚拟环境...
if exist "venv\Scripts\activate.bat" (
    echo        发现虚拟环境，正在激活...
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    echo        发现虚拟环境，正在激活...
    call .venv\Scripts\activate.bat
) else (
    echo        未发现虚拟环境，使用全局 Python 环境
)

:: ========== 检查 Streamlit ==========
echo [3/3] 检查 Streamlit...
python -c "import streamlit" >nul 2>&1
if %errorlevel% neq 0 (
    echo        Streamlit 未安装，正在安装依赖...
    echo.
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [错误] 依赖安装失败，请手动运行: pip install -r requirements.txt
        goto :error
    )
)
echo        Streamlit 已就绪

:: ========== 启动应用 ==========
echo.
echo ════════════════════════════════════════════════
echo   正在启动 Streamlit 应用...
echo   访问地址: http://localhost:8501
echo   按 Ctrl+C 停止服务
echo ════════════════════════════════════════════════
echo.

streamlit run Code/app.py --server.headless=true

goto :end

:error
echo.
echo [启动失败] 请检查上述错误信息
echo.
pause
exit /b 1

:end
echo.
echo 应用已停止运行
pause
