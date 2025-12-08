# 城市空气质量监测及归因预警系统

**项目代号**: AirQuality-StatModel-2025  
**课程**: 统计分析与建模  
**核心目标**: 构建一个基于多维统计推断与隐马尔可夫模型（HMM）的城市空气质量监测预警系统

---

## 📋 项目概述

本项目是一个完整的统计建模系统，使用**纯统计学方法**（非机器学习）分析城市空气质量数据，实现：

- **数据可视化与探索性分析**：正态性检验、分布拟合、ANOVA分析
- **统计推断**：假设检验、方差分析、相关分析
- **广义线性模型（GLM）**：使用Gamma分布族进行归因分析
- **时间序列分析**：ARIMA模型预测未来趋势
- **隐马尔可夫模型（HMM）**：推断空气质量隐状态

---

## 🎯 课程章节应用

本项目完整覆盖了《统计分析与建模》课程的13个章节：

### 1. 基础工具与理论
- ✅ **R语言思想**：使用statsmodels复现R的统计报表
- ✅ **数据可视化**：QQ图、热力图、时序分解图
- ✅ **数据预处理**：插值填补、3σ异常值剔除

### 2. 统计推断核心
- ✅ **分布拟合**：证明PM2.5服从Gamma/对数正态分布
- ✅ **假设检验**：T检验（周末vs工作日）
- ✅ **方差分析**：ANOVA（不同风向下的污染差异）
- ✅ **相关分析**：多重共线性检测

### 3. 回归与建模进阶
- ✅ **回归分析**：OLS作为Baseline
- ✅ **广义线性回归（GLM）**：Gamma分布族，核心模型
- ✅ **贝叶斯思想**：在报告中探讨先验后验更新

### 4. 高级时序与状态模型
- ✅ **时间序列分析**：ARIMA模型，捕捉季节性
- ✅ **HMM（隐马尔可夫模型）**：隐状态推断，核心亮点

---

## 📁 项目结构

```
Stat-Modeling/
├── Data/                    # 数据目录
│   └── PRSA_data.csv       # UCI Beijing PM2.5数据集（<10MB）
├── Code/                    # 代码目录
│   ├── app.py              # Streamlit主应用
│   ├── data_preprocessing.py    # 数据预处理模块
│   ├── statistical_inference.py # 统计推断模块
│   ├── glm_model.py        # GLM模型模块
│   ├── arima_model.py      # ARIMA模型模块
│   └── hmm_model.py        # HMM模型模块
├── Docs/                    # 文档目录
│   └── (项目报告、PPT等)
├── requirements.txt        # Python依赖
└── README.md              # 项目说明
```

---

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆仓库
git clone <your-repo-url>
cd Stat-Modeling

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

将UCI Beijing PM2.5数据集放置在 `Data/PRSA_data.csv`，或通过Streamlit界面上传。

**数据要求**：
- 必须包含 `PM2.5` 列
- 建议包含日期列（用于时间序列分析）
- 建议包含气象因子：`TEMP`, `PRES`, `DEWP`, `RAIN`, `WSPM`, `cbwd`等

### 3. 运行系统

```bash
# 启动Streamlit应用
streamlit run Code/app.py
```

浏览器将自动打开 `http://localhost:8501`

---

## 🎨 系统功能

### 页面A：数据洞察 (Data Insight)

- **PM2.5历史趋势图**：时间序列可视化
- **正态性检验**：
  - 直方图 vs 正态分布拟合曲线
  - Kolmogorov-Smirnov检验
  - 分布拟合（正态、Gamma、对数正态）
- **ANOVA分析**：
  - 不同风向下的PM2.5箱线图
  - F统计量、P值、各组统计量

### 页面B：归因分析 (Attribution Analysis)

- **特征选择**：交互式选择气象因子
- **多重共线性检测**：自动识别高相关变量对
- **GLM模型**：
  - 统计摘要表（P值、置信区间）
  - 显著特征高亮（P < 0.05）
  - 系数可视化与解释
  - 定量分析：系数含义解读

### 页面C：预警中心 (Warning Center)

- **HMM隐状态推断**：
  - 当前空气质量隐状态（优良/轻度污染/重度污染）
  - 状态概率分布
  - 状态转移矩阵可视化
- **ARIMA预测**：
  - 未来24小时PM2.5预测曲线
  - 95%置信区间
  - 预测统计量

---

## 📊 模型说明

### 1. 数据预处理

- **缺失值处理**：线性插值 + 前向/后向填充
- **异常值剔除**：3σ原则（均值±3倍标准差）
- **分布拟合**：Kolmogorov-Smirnov检验，AIC准则选择最佳分布

### 2. 统计推断

- **T检验**：独立样本t检验，检验周末vs工作日差异
- **ANOVA**：单因素方差分析，检验多组均值差异
- **相关分析**：Pearson/Spearman相关系数，多重共线性检测

### 3. GLM模型（Gamma分布族）

- **分布族**：Gamma分布（适合非负连续变量）
- **链接函数**：对数链接（log link）
- **输出**：统计摘要、系数解释、显著性检验

### 4. ARIMA模型

- **自动参数选择**：基于AIC准则
- **季节性**：支持SARIMA（季节性ARIMA）
- **预测**：点预测 + 置信区间

### 5. HMM模型

- **隐状态**：3状态（优良、轻度污染、重度污染）
- **观测值**：气象因子（温度、气压、湿度等）
- **输出**：状态转移矩阵、发射概率、当前状态推断

---

## 📈 评分标准对应

| 评分项 | 权重 | 本项目对应内容 |
|--------|------|---------------|
| 数据可视化 | 10% | 页面A：趋势图、直方图、箱线图、热力图 |
| 数据预处理与建模 | 20% | 分布拟合、GLM、ARIMA、HMM模型 |
| 模型解读与应用 | 20% | 系数解释、P值分析、定量分析 |
| 系统创新性 | 10% | HMM隐状态推断、多模型融合 |
| 系统演示 | 20% | Streamlit交互式界面、三个功能页面 |
| GitHub协作 | 10% | 标准仓库结构、代码规范 |
| 文档资料 | 10% | README、代码注释、项目报告 |

---

## 🔧 技术栈

- **Python 3.8+**
- **Streamlit**：Web应用框架
- **statsmodels**：统计建模（GLM、ARIMA）
- **scipy**：统计检验（t检验、ANOVA、KS检验）
- **hmmlearn**：隐马尔可夫模型
- **pandas/numpy**：数据处理
- **matplotlib/seaborn/plotly**：数据可视化

---

## 📝 使用示例

### 示例1：分布拟合

```python
from Code.data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor(df=df)
preprocessor.handle_missing_values()
preprocessor.remove_outliers(column='PM2.5')
dist_results = preprocessor.fit_distribution(column='PM2.5')
print(f"最佳拟合分布: {dist_results['best_fit']}")
```

### 示例2：GLM建模

```python
from Code.glm_model import GLMModel

glm = GLMModel(family='gamma', link='log')
glm.fit(X, y)
significant = glm.get_significant_features(alpha=0.05)
print(glm.interpret_coefficient('TEMP'))
```

### 示例3：HMM状态推断

```python
from Code.hmm_model import HMMModel

hmm = HMMModel(n_states=3)
hmm.fit(observations, pm25_values)
current_state = hmm.predict_current_state(current_obs)
print(f"当前状态: {current_state['state_name']}")
```

---

## ⚠️ 注意事项

1. **数据规模**：确保数据文件 < 10MB（UCI数据集约1.5MB，符合要求）
2. **模型类型**：本项目使用**纯统计学方法**，不使用机器学习（随机森林、神经网络等）
3. **统计严谨性**：所有模型都提供P值、置信区间、残差检验等统计指标
4. **代码规范**：遵循PEP 8，添加详细注释

---

## 📚 参考文献

- UCI Machine Learning Repository: Beijing PM2.5 Data Set
- statsmodels官方文档
- 《统计分析与建模》课程教材

---

## 👥 团队协作

本项目通过GitHub/Gitee进行版本控制和团队协作：

- **分支管理**：main（主分支）、dev（开发分支）
- **提交规范**：清晰的commit message
- **代码审查**：Pull Request机制

---

## 📄 许可证

本项目仅用于课程作业，请勿用于商业用途。

---

## 🎓 课程信息

**课程名称**：统计分析与建模  
**项目代号**：AirQuality-StatModel-2025  
**完成时间**：2025年

---

**祝项目顺利！** 🎉

