@echo off
REM MetaExtract 核对台 — 一键 demo 启动器（Windows）。
REM 双击即可运行；看这 12 篇样例不需要联网、不需要任何 API key。
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ==^> MetaExtract 核对台 demo 启动中...

REM 1) 找 Python 3.10+（优先用 py 启动器，其次 PATH 里的 python）。
set "PY="
for %%V in (3.13 3.12 3.11 3.10) do (
  if not defined PY ( py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V" )
)
if not defined PY (
  python -c "import sys; raise SystemExit(0 if sys.version_info[:2]>=(3,10) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo.
  echo X 没找到 Python 3.10 或更高版本。
  echo   请先安装 Python（务必勾选 "Add Python to PATH"^）：https://www.python.org/downloads/
  echo   装好后重新双击本文件即可。
  pause
  exit /b 1
)

REM 2) 首次运行创建隔离环境 .venv（只在本文件夹内）。
if not exist ".venv\" (
  echo ==^> 首次运行：创建隔离环境 .venv ...
  %PY% -m venv .venv || goto :fail
)

REM 3) 首次运行装依赖（需联网，约 1-3 分钟，仅这一次；之后秒开）。
if not exist ".venv\Scripts\streamlit.exe" (
  echo ==^> 首次运行：安装依赖（需联网，约 1-3 分钟，仅这一次）...
  ".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
  ".venv\Scripts\python.exe" -m pip install -e ".[app]" || goto :fail
)

REM 4) 跳过 Streamlit 首次运行的邮箱询问。
if not exist "%USERPROFILE%\.streamlit\" mkdir "%USERPROFILE%\.streamlit"
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
  > "%USERPROFILE%\.streamlit\credentials.toml" echo [general]
  >>"%USERPROFILE%\.streamlit\credentials.toml" echo email = ""
)

REM 5) 启动核对台。只读本地缓存——无需联网、无需 key。关掉本窗口即可停止。
echo ==^> 启动核对台，浏览器会自动打开（关掉本窗口即可停止）...
".venv\Scripts\streamlit.exe" run app\cockpit.py --server.headless=false --browser.gatherUsageStats=false
goto :end

:fail
echo.
echo X 出错了。请把上面的信息截图发给我。
pause
exit /b 1

:end
