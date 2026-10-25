#!/usr/bin/env bash
# MetaExtract 核对台 — 一键 demo 启动器（Mac / Linux）。
# 双击即可运行；看这 12 篇样例不需要联网、不需要任何 API key。
set -euo pipefail

# 双击启动时工作目录是用户主目录，所以先切到本脚本所在的文件夹。
cd "$(dirname "$0")"

pause_on_exit() {
  echo ""
  echo "✗ 出错了。把上面的文字截图发给我即可。"
  read -r -p "按回车键关闭…" _ || true
}
trap pause_on_exit ERR

echo "==> MetaExtract 核对台 demo 启动中…"

# 1) 找一个 Python 3.10+ 解释器。
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3 python; do
  if command -v "$c" >/dev/null 2>&1 \
     && "$c" -c 'import sys; raise SystemExit(0 if sys.version_info[:2]>=(3,10) else 1)' 2>/dev/null; then
    PY="$c"; break
  fi
done
if [ -z "$PY" ]; then
  trap - ERR
  echo ""
  echo "✗ 没找到 Python 3.10 或更高版本。"
  echo "  请先安装 Python（安装时若有 'Add to PATH' 请勾上）：https://www.python.org/downloads/"
  echo "  装好后重新双击本文件即可。"
  read -r -p "按回车键关闭…" _ || true
  exit 1
fi
echo "==> 使用 $("$PY" --version)"

# 2) 首次运行创建隔离环境 .venv（只在本文件夹内，不动系统 Python）。
if [ ! -d ".venv" ]; then
  echo "==> 首次运行：创建隔离环境 .venv …"
  "$PY" -m venv .venv
fi

# 3) 首次运行装依赖（需联网，约 1–3 分钟，仅这一次；之后秒开）。
if [ ! -x ".venv/bin/streamlit" ]; then
  echo "==> 首次运行：安装依赖（需联网，约 1–3 分钟，仅这一次）…"
  .venv/bin/python -m pip install --upgrade pip >/dev/null
  .venv/bin/python -m pip install -e ".[app]"
fi

# 4) 跳过 Streamlit 首次运行的邮箱询问，让浏览器直接打开。
if [ ! -f "$HOME/.streamlit/credentials.toml" ]; then
  mkdir -p "$HOME/.streamlit"
  printf '[general]\nemail = ""\n' > "$HOME/.streamlit/credentials.toml"
fi

# 5) 启动核对台。它只读本地缓存——无需联网、无需 key。关掉这个窗口即可停止。
echo "==> 启动核对台，浏览器会自动打开（关掉这个窗口即可停止）…"
trap - ERR
.venv/bin/streamlit run app/cockpit.py \
  --server.headless=false --browser.gatherUsageStats=false || {
    echo ""; echo "✗ 核对台启动失败，请把上面的信息发给我。"
    read -r -p "按回车键关闭…" _ || true; exit 1; }
