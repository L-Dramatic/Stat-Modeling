#!/bin/bash

# ============================================
# 城市空气质量监测预警系统 v2.0
# AirQuality-StatModel-2025
# ============================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     城市空气质量监测预警系统 v2.0              ║${NC}"
echo -e "${BLUE}║     AirQuality-StatModel-2025                  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════╝${NC}"
echo ""

# 切换到脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ========== 检查 Python 环境 ==========
echo -e "${YELLOW}[1/3]${NC} 检查 Python 环境..."
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo -e "${RED}[错误]${NC} 未检测到 Python，请先安装 Python 3.8+"
    echo "       Ubuntu/Debian: sudo apt install python3 python3-pip"
    echo "       macOS: brew install python3"
    exit 1
fi

# 优先使用 python3
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
else
    PYTHON_CMD="python"
    PIP_CMD="pip"
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo -e "       Python 版本: ${GREEN}$PYTHON_VERSION${NC}"

# ========== 检查虚拟环境 ==========
echo -e "${YELLOW}[2/3]${NC} 检查虚拟环境..."
if [ -f "venv/bin/activate" ]; then
    echo "       发现虚拟环境，正在激活..."
    source venv/bin/activate
    PYTHON_CMD="python"
    PIP_CMD="pip"
elif [ -f ".venv/bin/activate" ]; then
    echo "       发现虚拟环境，正在激活..."
    source .venv/bin/activate
    PYTHON_CMD="python"
    PIP_CMD="pip"
else
    echo "       未发现虚拟环境，使用全局 Python 环境"
fi

# ========== 检查 Streamlit ==========
echo -e "${YELLOW}[3/3]${NC} 检查 Streamlit..."
if ! $PYTHON_CMD -c "import streamlit" &> /dev/null; then
    echo "       Streamlit 未安装，正在安装依赖..."
    echo ""
    $PIP_CMD install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo -e "${RED}[错误]${NC} 依赖安装失败，请手动运行: $PIP_CMD install -r requirements.txt"
        exit 1
    fi
fi
echo -e "       Streamlit ${GREEN}已就绪${NC}"

# ========== 启动应用 ==========
echo ""
echo -e "${BLUE}════════════════════════════════════════════════${NC}"
echo -e "  正在启动 Streamlit 应用..."
echo -e "  访问地址: ${GREEN}http://localhost:8501${NC}"
echo -e "  按 ${YELLOW}Ctrl+C${NC} 停止服务"
echo -e "${BLUE}════════════════════════════════════════════════${NC}"
echo ""

# 启动 Streamlit
$PYTHON_CMD -m streamlit run Code/app.py --server.headless=true

echo ""
echo "应用已停止运行"
