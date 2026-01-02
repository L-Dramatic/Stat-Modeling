# 城市空气质量监测及归因分析系统

**项目代号**: AirQuality-StatModel-2025  
**版本**: V3.0 终极增强版  
**课程**: 统计分析与建模

## 📋 项目简介

基于多维统计推断的城市空气质量监测分析系统。使用纯统计学方法（非机器学习）分析PM2.5数据，实现**由浅入深**的完整统计建模流程：数据探索 → 统计推断 → 回归建模 → 分类预测。

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行系统

**方式一：命令行启动**
```bash
streamlit run Code/app.py
```

**方式二：使用启动脚本**
- Windows: 双击 `run_app.bat`
- Linux/Mac: 运行 `./run_app.sh`

浏览器将自动打开 `http://localhost:8501`

## 📁 项目结构

```
Stat-Modeling/
├── Code/                          # 核心代码模块
│   ├── __init__.py                # 包初始化文件
│   ├── app.py                     # Streamlit主应用（5个页面）
│   ├── data_preprocessing.py      # 数据预处理（插值、异常值、Log变换、正态性检验）
│   ├── statistical_inference.py   # 统计推断（T检验、ANOVA、相关分析）
│   ├── feature_selection.py       # 特征选择（Lasso）⭐ V3新增
│   ├── regression_models.py       # 回归模型（OLS/Ridge/Lasso/GLM）⭐ V3新增
│   ├── bayesian_models.py         # 贝叶斯模型 ⭐ V3新增
│   ├── classification_models.py   # 分类模型（Logistic/NB）⭐ V3新增
│   ├── model_evaluation.py        # 模型评估 ⭐ V3新增
│   └── glm_model.py               # 广义线性模型（Gamma分布族）
├── Data/                          # 数据目录
│   ├── beijing+pm2+5+data.zip     # UCI北京PM2.5数据集
│   └── README.md                  # 数据说明文档
├── Docs/                          # 文档目录
│   ├── 项目增强说明.md             # 功能增强详细说明
│   └── 项目报告模板.md             # 课程报告模板
├── README.md                      # 项目说明文档
├── requirements.txt               # Python依赖
├── run_app.bat                    # Windows启动脚本
└── run_app.sh                     # Linux/Mac启动脚本
```

## 🎨 系统功能

### 页面1：📊 数据洞察
- **概览指标**：总记录数、PM2.5均值/峰值、缺失值统计
- **历史趋势图**：交互式PM2.5浓度时序图（Plotly）
- **正态性检验**：分布拟合与检验
- **相关性热力图**：PM2.5与气象因子的相关系数可视化
- **T检验分析**：工作日 vs 周末 PM2.5差异显著性检验
- **ANOVA分析**：风向对PM2.5的影响分析

### 页面2：🔍 归因分析
- **OLS Baseline模型**：普通线性回归作为基准
- **GLM模型**：Gamma分布族 + Log链接函数
- **模型对比**：OLS vs GLM，展示模型选择过程
- **系数可视化**：条形图展示各因子影响（显著性颜色编码）
- **VIF共线性诊断**：多重共线性检测与预警

### 页面3：⚔️ 模型竞技场 ⭐ V3新增
- **特征选择**：Lasso特征筛选，识别关键变量
- **回归模型对比**：OLS vs Ridge vs Lasso vs GLM vs Bayesian Ridge
- **模型性能对比**：AIC/BIC、R²、RMSE、MAE对比表
- **残差分析**：残差分布图、Q-Q图、Durbin-Watson检验
- **贝叶斯方法**：参数后验分布可视化、可信区间

### 页面4：🎯 分类与状态 ⭐ V3新增
- **分类模型对比**：Logistic Regression vs Naive Bayes
- **混淆矩阵**：各模型的分类结果可视化
- **ROC曲线**：模型分类性能对比
- **评估指标**：Accuracy、Precision、Recall、F1-Score、AUC

### 页面5：📋 评估中心 ⭐ V3新增
- **统一模型评估**：所有模型的评估指标汇总
- **回归模型评估**：RMSE、MAE、R²、AIC、BIC
- **分类模型评估**：Accuracy、Precision、Recall、F1、AUC

## 📊 技术栈

- **Python 3.8+**
- **Streamlit** - Web应用框架
- **streamlit-option-menu** - 美化导航菜单
- **statsmodels** - 统计建模（OLS、Ridge、GLM）
- **scipy** - 统计检验（T检验、ANOVA、K-S检验）
- **scikit-learn** - Lasso、Ridge、Logistic、Naive Bayes
- **pandas/numpy** - 数据处理
- **matplotlib/seaborn/plotly** - 数据可视化

## 📈 课程章节覆盖（13个章节全覆盖）

### 基础与预处理
| 章节 | 实现内容 | 状态 |
|------|---------|------|
| **R语言** | 算法逻辑与R一致，附录提供R代码对照 | ✅ |
| **数据可视化** | 热力图、QQ图、残差图、ROC曲线 | ✅ |
| **数据预处理** | 插值填补、异常值剔除（3σ/IQR）、Log变换 | ✅ |
| **分布拟合** | 正态、Gamma、对数正态，KS检验、AIC选择 | ✅ |

### 统计推断
| 章节 | 实现内容 | 状态 |
|------|---------|------|
| **假设检验** | T检验（工作日vs周末） | ✅ |
| **方差分析** | ANOVA（风向影响） | ✅ |
| **相关分析** | Pearson/Spearman相关系数、VIF共线性检测 | ✅ |

### 回归建模
| 章节 | 实现内容 | 状态 |
|------|---------|------|
| **回归分析** | OLS线性回归（Baseline） | ✅ |
| **其他回归模型** | Ridge、Lasso（特征选择）⭐ V3新增 | ✅ |
| **贝叶斯方法** | Bayesian Ridge Regression，后验分布分析 ⭐ V3新增 | ✅ |
| **广义线性回归** | GLM（Gamma分布族 + Log链接） | ✅ |

### 分类
| 章节 | 实现内容 | 状态 |
|------|---------|------|
| **分类模型** | Logistic Regression、Naive Bayes ⭐ V3新增 | ✅ |

## 📝 数据要求

### 必需列
- `PM2.5`: PM2.5浓度值（μg/m³）

### 推荐列
- 日期/时间列：`date`、`year/month/day/hour`（用于时间序列分析）
- 气象因子：`TEMP`（温度）、`PRES`（气压）、`DEWP`（露点）、`RAIN`（降雨）
- 风速风向：`WSPM`/`Iws`（风速）、`cbwd`（风向，用于ANOVA）

### 数据来源
- **推荐数据集**: UCI Machine Learning Repository - Beijing PM2.5 Data Set
- **数据规模**: 约1.5MB，符合课程 <10MB 要求

## 🔧 预处理选项

系统支持以下预处理配置（可在侧边栏设置）：

| 选项 | 方法 | 说明 |
|------|------|------|
| 缺失值处理 | `interpolation` | 线性插值填补 |
| | `drop` | 直接删除缺失行 |
| 异常值处理 | `3sigma` | 3倍标准差原则 |
| | `iqr` | 四分位距原则 |
| | `none` | 不处理异常值 |
| Log变换 | 可选 | 使右偏数据正态化 |

## 📚 参考文献

- UCI Machine Learning Repository: Beijing PM2.5 Data Set
- statsmodels官方文档
- scikit-learn官方文档
- hmmlearn官方文档

---

**课程**: 统计分析与建模 | **版本**: V3.0 终极增强版 | **完成时间**: 2025年
