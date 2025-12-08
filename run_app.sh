#!/bin/bash

echo "===================================="
echo "城市空气质量监测预警系统"
echo "===================================="
echo ""
echo "正在启动Streamlit应用..."
echo ""

cd "$(dirname "$0")"
streamlit run Code/app.py

