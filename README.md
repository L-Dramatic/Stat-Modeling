# 城市空气质量监测及归因预警系统

**项目代号**: AirQuality-StatModel-2025  
**版本**: V3.0 终极增强版  
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
├── Code/                      # 核心代码模块
│   ├── app.py                 # Streamlit主应用（6个页面）
│   ├── data_preprocessing.py  # 数据预处理
│   ├── statistical_inference.py  # 统计推断
│   ├── feature_selection.py   # 特征选择（Lasso）⭐ 新增
│   ├── regression_models.py   # 回归模型（OLS/Ridge/Lasso/GLM）⭐ 新增
│   ├── bayesian_models.py     # 贝叶斯模型 ⭐ 新增
│   ├── classification_models.py  # 分类模型（Logistic/NB）⭐ 新增
│   ├── model_evaluation.py    # 模型评估 ⭐ 新增
│   ├── glm_model.py           # GLM模型
│   ├── arima_model.py         # ARIMA时间序列
│   └── hmm_model.py           # HMM隐马尔可夫模型
├── Data/                      # 数据目录
├── Docs/                      # 文档目录
│   └── 项目提案-终极增强版.md  # 完整项目提案 ⭐ 新增
└── requirements.txt           # Python依赖
```

## 🎨 系统功能

### 页面1：数据洞察
- PM2.5历史趋势图
- 正态性检验与分布拟合
- 相关分析热力图
- T检验（工作日vs周末）
- ANOVA分析（风向影响）

### 页面2：归因分析
- OLS Baseline模型
- GLM模型（Gamma分布族）
- 系数解释与显著性分析
- VIF多重共线性诊断

### 页面3：⚔️ 模型竞技场 ⭐ **新增**
- **特征选择**：Lasso特征筛选，识别关键变量
- **回归模型对比**：OLS vs Ridge vs Lasso vs GLM vs Bayesian Ridge
- **模型性能对比**：AIC/BIC、R²、RMSE、MAE对比表
- **残差分析**：残差分布图、Q-Q图、Durbin-Watson检验
- **贝叶斯方法**：参数后验分布可视化、可信区间

### 页面4：🎯 分类与状态 ⭐ **新增**
- **分类模型对比**：Logistic Regression vs Naive Bayes vs HMM
- **混淆矩阵**：各模型的分类结果可视化
- **ROC曲线**：模型分类性能对比
- **评估指标**：Accuracy、Precision、Recall、F1-Score、AUC

### 页面5：预警中心
- HMM隐状态推断
- ARIMA时间序列预测
- ADF平稳性检验

### 页面6：📋 评估中心 ⭐ **新增**
- 统一展示所有模型的评估指标
- 回归模型评估（RMSE、MAE、R²、AIC、BIC）
- 分类模型评估（Accuracy、Precision、Recall、F1、AUC）

## 📊 技术栈

- **Python 3.8+**
- **Streamlit** - Web应用框架
- **statsmodels** - 统计建模（GLM、ARIMA）
- **scipy** - 统计检验
- **hmmlearn** - 隐马尔可夫模型
- **pandas/numpy** - 数据处理
- **matplotlib/seaborn/plotly** - 数据可视化

## 📈 课程章节覆盖（13个章节全覆盖）

### 基础与预处理
- ✅ **R语言**：算法逻辑与R一致，附录提供R代码对照
- ✅ **数据可视化**：热力图、QQ图、残差图、ROC曲线
- ✅ **数据预处理**：插值、异常值剔除、Log变换
- ✅ **数据分布拟合**：正态、Gamma、对数正态，KS检验、AIC选择

### 统计推断
- ✅ **假设检验**：T检验（工作日vs周末）
- ✅ **方差分析**：ANOVA（风向影响）
- ✅ **相关分析**：相关系数矩阵、VIF多重共线性检测

### 回归建模
- ✅ **回归分析**：OLS基准模型
- ✅ **其他回归模型**：Ridge、Lasso（特征选择）⭐ 新增
- ✅ **贝叶斯方法**：Bayesian Ridge Regression，后验分布分析 ⭐ 新增
- ✅ **广义线性回归**：GLM（Gamma分布族）

### 分类与时序
- ✅ **分类模型**：Logistic Regression、Naive Bayes ⭐ 新增
- ✅ **时间序列分析**：ARIMA、ADF平稳性检验、时序分解
- ✅ **HMM**：隐马尔可夫模型（状态推断）

## 📝 数据要求

- 必须包含 `PM2.5` 列
- 建议包含日期列（用于时间序列分析）
- 建议包含气象因子：`TEMP`, `PRES`, `DEWP`, `RAIN`, `WSPM`, `cbwd`等

## 📚 参考文献

- UCI Machine Learning Repository: Beijing PM2.5 Data Set
- statsmodels官方文档

---

**课程**: 统计分析与建模 | **完成时间**: 2025年
