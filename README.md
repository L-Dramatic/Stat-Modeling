# 城市空气质量监测及归因预警系统

**项目代号**: AirQuality-StatModel-2025  
**课程**: 统计分析与建模

## 📋 项目简介

基于多维统计推断与隐马尔可夫模型（HMM）的城市空气质量监测预警系统。使用纯统计学方法（非机器学习）分析PM2.5数据，实现数据探索、归因分析和预警预测。

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行系统

```bash
streamlit run Code/app.py
```

浏览器将自动打开 `http://localhost:8501`

## 📁 项目结构

```
Stat-Modeling/
├── Code/              # 核心代码模块
│   ├── app.py         # Streamlit主应用
│   ├── data_preprocessing.py
│   ├── statistical_inference.py
│   ├── glm_model.py
│   ├── arima_model.py
│   └── hmm_model.py
├── Data/              # 数据目录
├── Docs/              # 文档目录
└── requirements.txt   # Python依赖
```

## 🎨 系统功能

### 页面A：数据洞察
- PM2.5历史趋势图
- 正态性检验与分布拟合
- 相关分析热力图
- T检验（工作日vs周末）
- ANOVA分析（风向影响）

### 页面B：归因分析
- OLS Baseline模型
- GLM模型（Gamma分布族）
- 系数解释与显著性分析

### 页面C：预警中心
- HMM隐状态推断
- ARIMA时间序列预测
- ADF平稳性检验

## 📊 技术栈

- **Python 3.8+**
- **Streamlit** - Web应用框架
- **statsmodels** - 统计建模（GLM、ARIMA）
- **scipy** - 统计检验
- **hmmlearn** - 隐马尔可夫模型
- **pandas/numpy** - 数据处理
- **matplotlib/seaborn/plotly** - 数据可视化

## 📈 课程章节覆盖

- ✅ 数据预处理（插值、异常值剔除）
- ✅ 分布拟合（正态、Gamma、对数正态）
- ✅ 假设检验（T检验、ANOVA）
- ✅ 相关分析（多重共线性检测）
- ✅ 回归分析（OLS、GLM）
- ✅ 时间序列分析（ARIMA）
- ✅ 隐马尔可夫模型（HMM）

## 📝 数据要求

- 必须包含 `PM2.5` 列
- 建议包含日期列（用于时间序列分析）
- 建议包含气象因子：`TEMP`, `PRES`, `DEWP`, `RAIN`, `WSPM`, `cbwd`等

## 📚 参考文献

- UCI Machine Learning Repository: Beijing PM2.5 Data Set
- statsmodels官方文档

---

**课程**: 统计分析与建模 | **完成时间**: 2025年
