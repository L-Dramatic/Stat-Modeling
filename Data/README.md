# 数据目录说明

## 数据要求

本项目使用 **UCI Beijing PM2.5数据集**，数据规模约1.5MB（<10MB，符合课程要求）。

## 数据来源

- **数据集名称**: Beijing PM2.5 Data Data Set
- **来源**: UCI Machine Learning Repository
- **下载链接**: https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data

## 数据格式

CSV格式，建议包含以下列：

### 必需列
- `PM2.5`: PM2.5浓度值（μg/m³）

### 推荐列
- `date` 或 `Date`: 日期时间（用于时间序列分析）
- `TEMP`: 温度（℃）
- `PRES`: 气压（hPa）
- `DEWP`: 露点温度（℃）
- `RAIN`: 降雨量（mm）
- `WSPM`: 风速（m/s）
- `cbwd`: 风向（分类变量，用于ANOVA分析）

## 数据预处理

系统会自动进行以下预处理：
1. 缺失值插值填补
2. 异常值剔除（3σ原则）
3. 分布拟合检验

## 使用说明

1. 将数据文件命名为 `PRSA_data.csv` 并放在此目录下
2. 或通过Streamlit界面上传数据文件
3. 系统会自动识别数据格式并进行处理

## 注意事项

- 确保数据文件大小 < 10MB
- 日期列建议使用标准格式（YYYY-MM-DD HH:MM:SS）
- 缺失值不要超过总数据的20%

