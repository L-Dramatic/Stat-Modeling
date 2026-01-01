import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import statsmodels.api as sm
import sys
import os

# 尝试导入美化菜单库，如果没有安装则降级处理
try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except ImportError:
    HAS_OPTION_MENU = False

# 设置matplotlib和seaborn的全局样式
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 设置matplotlib中文字体支持（解决中文乱码问题）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 添加Code目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入你的模块
try:
    from data_preprocessing import DataPreprocessor
    from statistical_inference import StatisticalInference
    from glm_model import GLMModel
    from arima_model import ARIMAModel
    from hmm_model import HMMModel
    # 新增模块
    from classification_models import ClassificationModels
    from model_evaluation import ModelEvaluator
    from bayesian_models import BayesianModels
    from regression_models import RegressionModels
    from feature_selection import FeatureSelector
except ImportError as e:
    st.warning(f"部分模块导入失败: {e}")
    pass

# ==========================================
# 1. 页面配置与 CSS 美化
# ==========================================
st.set_page_config(
    page_title="空气质量监测预警系统",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 注入自定义 CSS (这是变好看的关键)
st.markdown("""
<style>
    /* 全局字体与背景 */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }
    
    /* 主背景色微调 */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* 标题样式增强 */
    h1 {
        color: #2c3e50;
        font-weight: 700 !important;
        letter-spacing: -1px;
    }
    h2, h3 {
        color: #34495e;
        font-weight: 600 !important;
    }
    
    /* 卡片式容器样式 */
    .css-card {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        border: 1px solid #e9ecef;
    }
    
    /* 指标 (Metric) 样式优化 */
    div[data-testid="stMetric"] {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #eee;
        text-align: center;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #7f8c8d;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #2980b9;
    }

    /* 侧边栏优化 */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #eee;
    }
    
    /* 按钮美化 */
    .stButton > button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #2980b9;
        box-shadow: 0 4px 8px rgba(52, 152, 219, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心逻辑函数 (逻辑保持不变)
# ==========================================

def get_hmm_features(df):
    # 你这份 UCI 数据的典型气象列
    candidate = ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
    feats = [c for c in candidate if c in df.columns]
    return feats


def normalize_column_names(df):
    df = df.copy()
    column_mapping = {}
    pm25_variants = ['PM2.5','pm2.5','PM2_5','pm2_5','PM25','pm25','PM 2.5','pm 2.5']

    pm25_col = None
    for col in df.columns:
        if col in pm25_variants or col.strip() in pm25_variants:
            pm25_col = col
            break
    if pm25_col and pm25_col != 'PM2.5':
        df.rename(columns={pm25_col: 'PM2.5'}, inplace=True)

    date_variants = ['date','Date','DATE','datetime','DateTime','DATETIME','time','Time','timestamp','utc_time']
    date_col = None
    for col in df.columns:
        if col in date_variants:
            date_col = col
            break
    if date_col and date_col != 'date':
        df.rename(columns={date_col: 'date'}, inplace=True)

    return df, column_mapping


@st.cache_data
def load_data(file_path):
    """加载数据 (缓存)"""
    try:
        if isinstance(file_path, str):
            df = pd.read_csv(file_path, na_values=['NA', 'NaN', '?', 'null'])
        else:
            # 如果是上传的文件对象，需要重置指针
            if hasattr(file_path, 'seek'):
                file_path.seek(0)
            df = pd.read_csv(file_path, na_values=['NA', 'NaN', '?', 'null'])
        
        df, mapping = normalize_column_names(df)
        lower_map = {c.lower(): c for c in df.columns}
        
        if 'date' not in df.columns:
            required_cols = ['year', 'month', 'day']
            if all(key in lower_map for key in required_cols):
                try:
                    datetime_parts = {
                        'year': df[lower_map['year']],
                        'month': df[lower_map['month']],
                        'day': df[lower_map['day']]
                    }
                    if 'hour' in lower_map:
                        datetime_parts['hour'] = df[lower_map['hour']]
                    
                    df['date'] = pd.to_datetime(datetime_parts, errors='coerce')
                    df = df.dropna(subset=['date']).set_index('date').sort_index()
                except:
                    pass
        else:
            try:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.dropna(subset=['date']).set_index('date').sort_index()
            except:
                pass
        
        if 'PM2.5' not in df.columns:
            return None
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None

@st.cache_data
def preprocess_data(df, missing_method="interpolation", outlier_method="3sigma", do_log=False):
    """数据预处理 (缓存) - 调用 DataPreprocessor"""
    if 'PM2.5' not in df.columns:
        return df

    pre = DataPreprocessor(df=df)

    # 1) 缺失值
    pre.handle_missing_values(method=missing_method)

    # 2) 异常值
    if outlier_method != "none":
        pre.remove_outliers(column="PM2.5", method=outlier_method)

    df_processed = pre.get_processed_data()

    # 3) log 变换：不替换原列，只增加一列方便对比
    if do_log:
        try:
            df_processed["log_PM2.5"] = pre.log_transform("PM2.5")
        except:
            pass

    return df_processed


# ==========================================
# 3. 页面视图函数 (UI 升级版)
# ==========================================

def page_data_insight(df):
    st.markdown("## 📊 数据全景洞察")
    
    # 使用容器包裹，增加间距
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📚 总记录数", f"{len(df):,}")
        col2.metric("🌫️ PM2.5 均值", f"{df['PM2.5'].mean():.1f}")
        col3.metric("📈 PM2.5 峰值", f"{df['PM2.5'].max():.1f}")
        col4.metric("📉 当前缺失值", df['PM2.5'].isna().sum())
    
    st.markdown("---")

    # 图表区域 1
    col_chart1, col_chart2 = st.columns([2, 1])
    
    with col_chart1:
        st.markdown("### 📅 历史趋势回溯")
        if isinstance(df.index, pd.DatetimeIndex):
            fig = go.Figure()
            # 降采样防止卡顿
            plot_df = df.resample('D').mean(numeric_only=True) if len(df) > 10000 else df
            
            fig.add_trace(go.Scatter(
                x=plot_df.index, y=plot_df['PM2.5'],
                mode='lines', name='PM2.5',
                line=dict(color='#3498db', width=2),
                fill='tozeroy', fillcolor='rgba(52, 152, 219, 0.1)' 
            ))
            fig.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                height=400, template='plotly_white',
                xaxis_title="", yaxis_title="PM2.5 浓度 (μg/m³)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ 数据未包含时间索引，无法绘制趋势图")

    with col_chart2:
        st.markdown("### 🌡️ 相关性热力图")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cols = [c for c in numeric_cols if c not in ['No', 'year', 'month', 'day', 'hour', 'is_weekend']]
        if len(cols) > 1:
            corr = df[cols].corr()
            fig, ax = plt.subplots(figsize=(5, 6))
            sns.heatmap(corr[['PM2.5']].sort_values(by='PM2.5', ascending=False), 
                        annot=True, fmt='.2f', cmap='coolwarm', cbar=False, ax=ax)
            ax.set_ylabel('')
            st.pyplot(fig)

    st.markdown("---")
    
    # 更多分析折叠起来
    with st.expander("🧐 查看更多统计检验 (工作日效应 & 周期性)"):
        if isinstance(df.index, pd.DatetimeIndex):
            df_wk = df.copy()
            
            # === [修复点在这里] ===
            # 原代码报错：AttributeError: 'numpy.ndarray' object has no attribute 'map'
            # 修复方案：使用 np.where，如果>=5则是周末，否则是工作日
            df_wk['Type'] = np.where(df_wk.index.dayofweek >= 5, '周末', '工作日')
            # =====================
            
            col_ex1, col_ex2 = st.columns(2)
            with col_ex1:
                st.markdown("**工作日 vs 周末分布**")
                fig, ax = plt.subplots(figsize=(6, 4))
                # 指定 order 确保顺序一致
                sns.boxplot(data=df_wk, x='Type', y='PM2.5', palette="Set2", ax=ax, order=['工作日', '周末'])
                st.pyplot(fig)
            with col_ex2:
                st.markdown("**统计显著性 (T-Test)**")
                try:
                    inference = StatisticalInference(df_wk)
                    # 注意：这里需要传入 0/1 的数值列给 t_test 计算，因为 StatisticalInference 可能不支持中文标签列计算
                    # 所以我们临时加一个数值标识
                    df_wk['is_weekend_num'] = (df_wk.index.dayofweek >= 5).astype(int)
                    
                    res = inference.t_test('PM2.5', 'is_weekend_num') 
                    st.info(f"P-Value: **{res.get('p_value', 0):.4f}**")
                    if res.get('significant'):
                        st.success("✅ 差异显著：人类活动对空气质量有明显影响")
                    else:
                        st.warning("⚠️ 差异不显著")
                except Exception as e:
                    st.write(f"计算统计量时出错: {e}")

def compute_vif(X_df):
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X = sm.add_constant(X_df).dropna()
    vifs = []
    for i in range(X.shape[1]):
        vifs.append(variance_inflation_factor(X.values, i))
    return pd.DataFrame({"feature": X.columns, "VIF": vifs}).sort_values("VIF", ascending=False)


def page_attribution_analysis(df):
    st.markdown("## 🔍 归因分析实验室")
    st.info("💡 通过统计模型量化各个气象因子对 PM2.5 的具体贡献度。")
    
    col_ctrl, col_res = st.columns([1, 3])
    
    with col_ctrl:
        st.markdown("#### ⚙️ 参数配置")
        with st.form("model_params"):
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            features = [c for c in numeric_cols if c not in ['PM2.5', 'No', 'year', 'month', 'day', 'hour']]
            
            selected_features = st.multiselect("选择特征变量", features, default=features[:4] if features else [])
            model_type = st.radio("模型选择", ["OLS (线性回归)", "GLM (Gamma分布)"])
            
            submit = st.form_submit_button("🚀 开始建模")
    
    with col_res:
        if submit and selected_features:
            with st.spinner("正在拟合模型..."):
                try:
                    import statsmodels.api as sm
                    X_raw = df[selected_features]
                    X = sm.add_constant(X_raw).dropna()
                    y = df.loc[X.index, 'PM2.5']

                    if "OLS" in model_type:
                        model = sm.OLS(y, X).fit()
                        title = "OLS 线性回归结果"
                        coefs = model.params.drop('const', errors='ignore')
                        pvals = model.pvalues.drop('const', errors='ignore')

                        st.markdown(f"#### 📊 {title}")

                        fig, ax = plt.subplots(figsize=(10, 4))
                        colors = ['#2ecc71' if p < 0.05 else '#95a5a6' for p in pvals]
                        coefs.plot(kind='bar', color=colors, ax=ax)
                        ax.set_title("Feature Coefficients (green = significant)", fontsize=10)
                        ax.axhline(0, color='black', linewidth=0.8)
                        st.pyplot(fig)

                        with st.expander("📄 查看详细统计报表"):
                            st.text(model.summary())

                    else:
                        # ✅ 用你的 GLMModel
                        glm = GLMModel()
                        glm.fit(X_raw.dropna(), y.loc[X_raw.dropna().index])
                        title = "GLM (Gamma + log link) 结果"

                        st.markdown(f"#### 📊 {title}")

                        sig = glm.get_significant_features(alpha=0.05)
                        coefs = glm.results.params.drop('const', errors='ignore')
                        pvals = glm.results.pvalues.drop('const', errors='ignore')

                        fig, ax = plt.subplots(figsize=(10, 4))
                        colors = ['#2ecc71' if p < 0.05 else '#95a5a6' for p in pvals]
                        coefs.plot(kind='bar', color=colors, ax=ax)
                        ax.set_title("GLM Coefficients (Gamma + log link)", fontsize=10)
                        ax.axhline(0, color='black', linewidth=0.8)
                        st.pyplot(fig)

                        with st.expander("📌 显著因子解释（相对变化%）", expanded=True):
                            if sig.empty:
                                st.warning("没有显著因子（p<0.05）")
                            else:
                                for feat in sig.index:
                                    st.write(glm.interpret_coefficient(feat))

                        with st.expander("📄 查看 GLM 统计报表"):
                            st.text(glm.get_summary())

                    # ✅ VIF 共线性
                    with st.expander("🧪 多重共线性诊断（VIF）", expanded=False):
                        vif_df = compute_vif(df[selected_features])
                        st.dataframe(vif_df, use_container_width=True)
                        high_vif = vif_df[vif_df["VIF"] > 10]
                        if len(high_vif) > 0:
                            st.warning("以下变量 VIF>10，共线性较强，建议删减或合并：")
                            st.write(high_vif)

                        
                except Exception as e:
                    st.error(f"建模失败: {str(e)}")
        elif not submit:
            st.markdown("""
            <div style="text-align: center; padding: 50px; color: #95a5a6;">
                👈 请在左侧选择特征并点击运行
            </div>
            """, unsafe_allow_html=True)

def page_warning_center(df):
    st.markdown("## ⚠️ 智能预警中心")

    if not isinstance(df.index, pd.DatetimeIndex):
        st.error("当前数据没有时间索引，无法进行 HMM/ARIMA 预警。")
        return

    # =======================
    # 1) HMM 状态识别
    # =======================
    st.markdown("### 🎲 状态识别 (HMM)")
    feats = get_hmm_features(df)

    if len(feats) == 0:
        st.warning("未找到气象特征列（DEWP/TEMP/PRES/Iws/Is/Ir），无法拟合 HMM。")
    else:
        col1, col2 = st.columns([1, 2])

        with col1:
            n_states = st.slider("隐状态数量", 2, 5, 3)
            hmm_mode = st.radio("状态定义方式", ["阈值（国标）", "分位数"], index=0)

            run_hmm = st.button("🚀 拟合 HMM 并推断当前状态", use_container_width=True)

        with col2:
            st.markdown("**HMM 观测特征：** " + ", ".join(feats))

        if run_hmm:
            with st.spinner("HMM 训练中..."):
                obs = df[feats].dropna()
                pm25 = df.loc[obs.index, "PM2.5"].dropna()
                obs = obs.loc[pm25.index]

                hmm_model = HMMModel(n_states=n_states)

                # 用 PM2.5 来定义 state 的阈值/分位数（在模型里）
                hmm_model.fit(obs.values, pm25_values=pm25.values)

                # 推断全序列状态
                states = hmm_model.predict_states(obs.values)

                # ✅ 对齐状态含义：按每个 state 的 PM2.5 均值排序
                state_means = {}
                for s in range(n_states):
                    state_means[s] = pm25.values[states == s].mean()

                sorted_states = sorted(state_means, key=state_means.get)
                mapped_names = []
                if n_states == 3 and hmm_mode.startswith("阈值"):
                    mapped_names = ["优良", "轻度污染", "重度污染"]
                else:
                    mapped_names = [f"状态{i+1}" for i in range(n_states)]

                mapping = {s: mapped_names[i] for i, s in enumerate(sorted_states)}
                current_state = mapping[states[-1]]

                st.success(f"当前隐状态：**{current_state}**")
                mean_df = pd.DataFrame({
                    "state": list(state_means.keys()),
                    "PM2.5_mean": list(state_means.values())
                }).sort_values("PM2.5_mean")
                st.markdown("**各状态 PM2.5 均值（用于解释对齐）：**")
                st.dataframe(mean_df, use_container_width=True)

                st.markdown("#### 🔁 状态转移矩阵")
                trans = hmm_model.get_transition_matrix().copy()
                # 重新按 mapping 排序/重命名
                trans.index = [mapping.get(i, i) for i in trans.index]
                trans.columns = [mapping.get(i, i) for i in trans.columns]
                st.dataframe(trans, use_container_width=True)

                st.markdown("#### 📌 最近 7 天隐状态序列")
                last_idx = obs.index[-24*7:] if len(obs) >= 24*7 else obs.index
                last_states = states[-len(last_idx):]
                state_series = pd.Series([mapping[s] for s in last_states], index=last_idx)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=state_series.index, y=state_series.values, mode="lines"))
                fig.update_layout(height=250, margin=dict(t=20,b=0,l=0,r=0))
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # =======================
    # 2) ARIMA 短期预测
    # =======================
    st.markdown("### 🔮 趋势预测 (ARIMA)")

    col_arima1, col_arima2 = st.columns([2, 1])
    with col_arima1:
        steps = st.slider("预测未来小时数", 12, 72, 24)
    with col_arima2:
        use_auto_select = st.checkbox("自动选择参数（较慢）", value=False, help="取消勾选将使用默认参数(1,1,1)，速度更快")
    
    run_arima = st.button("📈 生成 ARIMA 预测", type="primary", use_container_width=True)

    if run_arima:
        series = df["PM2.5"].dropna()
        
        # 如果数据量太大，提示降采样
        if len(series) > 10000:
            st.info(f"💡 数据量较大（{len(series)}条），为加快速度将自动降采样")
            # 降采样到最近10000条
            series = series.iloc[-10000:]
        
        arima = ARIMAModel()

        # 平稳性检验
        with st.spinner("正在进行平稳性检验..."):
            stat_res = arima.check_stationarity(series)
            st.write("ADF 检验结果：", stat_res)

        # 拟合
        if use_auto_select:
            with st.spinner("正在自动选择ARIMA参数（这可能需要1-2分钟，请耐心等待）..."):
                arima.fit(series, auto_select=True)
                st.success(f"✅ 自动选择参数：ARIMA{arima.order}")
        else:
            with st.spinner("正在拟合ARIMA模型（使用默认参数(1,1,1)）..."):
                arima.fit(series, auto_select=False, order=(1, 1, 1))
                st.success("✅ 使用默认参数：ARIMA(1,1,1)")

        # 预测（两种模式都需要执行）
        with st.spinner("正在生成预测..."):
            forecast_df = arima.predict(steps=steps, alpha=0.05)

        st.markdown("#### 📊 预测曲线（含95%置信区间）")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values, mode="lines", name="历史 PM2.5"
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df.index, y=forecast_df["forecast"],
            mode="lines+markers", name="预测"
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df.index, y=forecast_df["upper"],
            mode="lines", name="上界", line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df.index, y=forecast_df["lower"],
            mode="lines", name="下界", fill="tonexty",
            line=dict(width=0), showlegend=False
        ))
        fig.update_layout(height=350, margin=dict(t=20,b=0,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📄 ARIMA 模型摘要"):
            st.text(arima.get_summary())


def page_model_arena(df):
    """模型竞技场页面 - 回归模型对比"""
    st.markdown("## ⚔️ 模型竞技场")
    st.info("💡 对比不同回归模型的性能，展示模型选择过程。")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in ['PM2.5', 'No', 'year', 'month', 'day', 'hour']]
    
    if len(features) == 0:
        st.warning("未找到可用的特征变量")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### ⚙️ 参数配置")
        selected_features = st.multiselect("选择特征变量", features, default=features[:4] if len(features) >= 4 else features)
        use_lasso_selection = st.checkbox("使用Lasso进行特征筛选", value=False)
        run_models = st.button("🚀 运行所有模型", type="primary", use_container_width=True)
    
    with col2:
        if run_models and selected_features:
            with st.spinner("正在拟合模型并计算评估指标..."):
                try:
                    X_raw = df[selected_features].dropna()
                    y = df.loc[X_raw.index, 'PM2.5'].dropna()
                    X_raw = X_raw.loc[y.index]
                    
                    # 特征选择
                    selected_X = X_raw
                    if use_lasso_selection:
                        selector = FeatureSelector()
                        result = selector.lasso_selection(X_raw, y)
                        selected_X = X_raw[result['selected_features']]
                        st.success(f"Lasso筛选出 {result['n_selected']}/{result['n_total']} 个重要特征")
                        
                        fig, ax = selector.plot_feature_importance(top_n=min(10, len(selected_features)))
                        st.pyplot(fig)
                    
                    # 拟合多个模型
                    reg_models = RegressionModels()
                    evaluator = ModelEvaluator()
                    
                    models_results = {}
                    
                    # OLS
                    ols_model = reg_models.fit_ols(selected_X, y)
                    y_pred_ols = ols_model.predict(sm.add_constant(selected_X))
                    models_results['OLS'] = {
                        'y_true': y,
                        'y_pred': y_pred_ols,
                        'model': ols_model
                    }
                    
                    # Ridge
                    ridge_model = reg_models.fit_ridge(selected_X, y, cv=True)
                    y_pred_ridge = ridge_model.predict(reg_models.scaler.transform(selected_X.values))
                    models_results['Ridge'] = {
                        'y_true': y,
                        'y_pred': y_pred_ridge,
                        'model': ridge_model
                    }
                    
                    # Lasso
                    lasso_model = reg_models.fit_lasso(selected_X, y, cv=True)
                    y_pred_lasso = lasso_model.predict(reg_models.scaler.transform(selected_X.values))
                    models_results['Lasso'] = {
                        'y_true': y,
                        'y_pred': y_pred_lasso,
                        'model': lasso_model
                    }
                    
                    # GLM
                    glm_model = reg_models.fit_glm(selected_X, y)
                    y_pred_glm = glm_model.predict(selected_X)
                    models_results['GLM'] = {
                        'y_true': y,
                        'y_pred': y_pred_glm,
                        'model': glm_model.results
                    }
                    
                    # Bayesian Ridge
                    bayesian = BayesianModels()
                    bayesian.fit_bayesian_regression(selected_X, y)
                    y_pred_bayesian, y_std = bayesian.predict_bayesian_regression(selected_X)
                    models_results['Bayesian Ridge'] = {
                        'y_true': y,
                        'y_pred': y_pred_bayesian,
                        'model': bayesian.bayesian_ridge_model
                    }
                    
                    # 模型对比
                    st.markdown("#### 📊 模型性能对比")
                    comparison_df = evaluator.compare_models(models_results, metric_type='regression')
                    st.dataframe(comparison_df.style.highlight_max(axis=0, subset=['R²']).highlight_min(axis=0, subset=['AIC', 'BIC', 'RMSE', 'MAE']), use_container_width=True)
                    
                    # 保存评估结果到session_state（供评估中心页面使用）
                    # 注意：只保存数据和对比表格，不保存模型对象和evaluator
                    st.session_state['regression_evaluation'] = {
                        'comparison_df': comparison_df,
                        'models_results': {k: {
                            'y_true': np.array(v['y_true']).flatten(),
                            'y_pred': np.array(v['y_pred']).flatten()
                        } for k, v in models_results.items()},
                        'selected_features': selected_X.columns.tolist()
                    }
                    
                    # 残差分析
                    st.markdown("#### 📈 残差分析")
                    model_choice = st.selectbox("选择模型查看残差", list(models_results.keys()))
                    if model_choice:
                        fig = evaluator.plot_residuals(
                            models_results[model_choice]['y_true'],
                            models_results[model_choice]['y_pred']
                        )
                        st.pyplot(fig)
                        
                        # Durbin-Watson检验
                        dw_result = evaluator.durbin_watson_test(
                            models_results[model_choice]['y_true'] - models_results[model_choice]['y_pred']
                        )
                        st.info(f"Durbin-Watson统计量: {dw_result['dw_statistic']:.4f} - {dw_result['interpretation']}")
                    
                    # 贝叶斯后验分布
                    st.markdown("#### 🎲 贝叶斯方法：参数后验分布")
                    fig, ax = bayesian.plot_posterior(feature_names=selected_X.columns.tolist())
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"模型拟合失败: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())


def page_classification(df):
    """分类与状态页面 - 分类模型对比"""
    st.markdown("## 🎯 分类与状态")
    st.info("💡 对比Logistic Regression、Naive Bayes和HMM的分类性能。")
    
    if 'PM2.5' not in df.columns:
        st.error("数据中未找到PM2.5列")
        return
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in ['PM2.5', 'No', 'year', 'month', 'day', 'hour']]
    
    if len(features) == 0:
        st.warning("未找到可用的特征变量")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### ⚙️ 参数配置")
        selected_features = st.multiselect("选择特征变量", features, default=features[:4] if len(features) >= 4 else features)
        run_classification = st.button("🚀 运行分类模型", type="primary", use_container_width=True)
    
    with col2:
        if run_classification and selected_features:
            with st.spinner("正在训练分类模型..."):
                try:
                    X = df[selected_features].dropna()
                    y_pm25 = df.loc[X.index, 'PM2.5'].dropna()
                    X = X.loc[y_pm25.index]
                    
                    # 初始化分类模型
                    clf_models = ClassificationModels()
                    evaluator = ModelEvaluator()
                    
                    # Logistic Regression
                    clf_models.fit_logistic(X, pm25_values=y_pm25.values)
                    y_pred_logistic = clf_models.predict_logistic(X)
                    y_proba_logistic = clf_models.predict_proba_logistic(X)
                    
                    # Naive Bayes
                    clf_models.fit_naive_bayes(X, pm25_values=y_pm25.values)
                    y_pred_nb = clf_models.predict_naive_bayes(X)
                    y_proba_nb = clf_models.predict_proba_naive_bayes(X)
                    
                    # 转换为分类标签（用于评估）
                    y_true = clf_models._pm25_to_category(y_pm25.values)
                    
                    # 评估
                    eval_logistic = clf_models.evaluate(y_true, y_pred_logistic, y_proba_logistic, "Logistic Regression")
                    eval_nb = clf_models.evaluate(y_true, y_pred_nb, y_proba_nb, "Naive Bayes")
                    
                    # HMM（使用现有HMM模块）
                    hmm_feats = get_hmm_features(df)
                    if len(hmm_feats) > 0:
                        hmm_obs = df[hmm_feats].loc[X.index].dropna()
                        hmm_pm25 = y_pm25.loc[hmm_obs.index]
                        hmm_obs = hmm_obs.loc[hmm_pm25.index]
                        
                        hmm_model = HMMModel(n_states=3)
                        hmm_model.fit(hmm_obs.values, pm25_values=hmm_pm25.values)
                        hmm_states = hmm_model.predict_states(hmm_obs.values)
                        
                        # 对齐HMM状态和分类标签
                        state_means = {}
                        for s in range(3):
                            state_means[s] = hmm_pm25.values[hmm_states == s].mean() if np.sum(hmm_states == s) > 0 else 0
                        sorted_states = sorted(state_means, key=state_means.get)
                        state_mapping = {sorted_states[i]: i for i in range(3)}
                        hmm_labels = np.array([state_mapping[s] for s in hmm_states])
                        y_true_hmm = clf_models._pm25_to_category(hmm_pm25.values)
                        eval_hmm = evaluator.classification_metrics(y_true_hmm, hmm_labels)
                    else:
                        eval_hmm = None
                    
                    # 展示结果
                    st.markdown("#### 📊 分类模型性能对比")
                    comparison_data = {
                        'Logistic Regression': {
                            'Accuracy': eval_logistic['accuracy'],
                            'Precision (Macro)': eval_logistic['precision_macro'],
                            'Recall (Macro)': eval_logistic['recall_macro'],
                            'F1-Score (Macro)': eval_logistic['f1_macro'],
                            'AUC': eval_logistic['auc_score']
                        },
                        'Naive Bayes': {
                            'Accuracy': eval_nb['accuracy'],
                            'Precision (Macro)': eval_nb['precision_macro'],
                            'Recall (Macro)': eval_nb['recall_macro'],
                            'F1-Score (Macro)': eval_nb['f1_macro'],
                            'AUC': eval_nb['auc_score']
                        }
                    }
                    
                    if eval_hmm:
                        comparison_data['HMM'] = {
                            'Accuracy': eval_hmm['accuracy'],
                            'Precision (Macro)': eval_hmm['precision_macro'],
                            'Recall (Macro)': eval_hmm['recall_macro'],
                            'F1-Score (Macro)': eval_hmm['f1_macro'],
                            'AUC': eval_hmm['auc_score']
                        }
                    
                    comparison_df = pd.DataFrame(comparison_data).T
                    st.dataframe(comparison_df.style.highlight_max(axis=0), use_container_width=True)
                    
                    # 保存评估结果到session_state（供评估中心页面使用）
                    st.session_state['classification_evaluation'] = {
                        'comparison_df': comparison_df,
                        'y_true': np.array(y_true).flatten(),
                        'y_pred_logistic': np.array(y_pred_logistic).flatten(),
                        'y_pred_nb': np.array(y_pred_nb).flatten(),
                        'y_proba_logistic': np.array(y_proba_logistic),
                        'y_proba_nb': np.array(y_proba_nb),
                        'eval_logistic': eval_logistic,
                        'eval_nb': eval_nb,
                        'eval_hmm': eval_hmm,
                        'class_names': clf_models.get_class_names()
                    }
                    
                    # 混淆矩阵对比
                    col_cm1, col_cm2 = st.columns(2)
                    with col_cm1:
                        st.markdown("**Logistic Regression 混淆矩阵**")
                        fig, ax = evaluator.plot_confusion_matrix(y_true, y_pred_logistic, clf_models.get_class_names())
                        st.pyplot(fig)
                    
                    with col_cm2:
                        st.markdown("**Naive Bayes 混淆矩阵**")
                        fig, ax = evaluator.plot_confusion_matrix(y_true, y_pred_nb, clf_models.get_class_names())
                        st.pyplot(fig)
                    
                    # ROC曲线对比
                    st.markdown("#### 📈 ROC曲线对比")
                    fig, ax = evaluator.plot_roc_curve(y_true, y_proba_logistic, clf_models.get_class_names())
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"分类模型训练失败: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())


def page_evaluation_center(df):
    """评估中心页面 - 统一评估所有模型"""
    st.markdown("## 📋 评估中心")
    st.info("💡 统一展示所有模型的评估指标和性能对比。")
    
    # 检查是否有评估结果
    has_regression = 'regression_evaluation' in st.session_state
    has_classification = 'classification_evaluation' in st.session_state
    
    if not has_regression and not has_classification:
        st.warning("""
        **⚠️ 暂无评估结果**
        
        请先在其他页面运行模型：
        - **回归模型**：前往"⚔️ 模型竞技场"页面，选择特征并运行所有模型
        - **分类模型**：前往"🎯 分类与状态"页面，选择特征并运行分类模型
        
        运行后，评估结果将自动显示在这里。
        """)
        return
    
    # =======================
    # 1. 回归模型评估
    # =======================
    st.markdown("### 📊 回归模型评估")
    
    if has_regression:
        reg_eval = st.session_state['regression_evaluation']
        comparison_df = reg_eval['comparison_df']
        models_results = reg_eval['models_results']
        evaluator = ModelEvaluator()  # 重新创建evaluator
        
        # 评估指标对比表
        st.markdown("#### 📈 模型性能对比表")
        st.dataframe(
            comparison_df.style.highlight_max(axis=0, subset=['R²'])
                          .highlight_min(axis=0, subset=['AIC', 'BIC', 'RMSE', 'MAE']),
            use_container_width=True
        )
        
        # 模型选择建议
        st.markdown("#### 💡 模型选择建议")
        best_r2 = comparison_df['R²'].idxmax()
        best_aic = comparison_df['AIC'].idxmin() if 'AIC' in comparison_df.columns and comparison_df['AIC'].notna().any() else None
        
        col_rec1, col_rec2 = st.columns(2)
        with col_rec1:
            st.success(f"**最佳R²模型**：{best_r2} (R² = {comparison_df.loc[best_r2, 'R²']:.4f})")
        with col_rec2:
            if best_aic:
                st.success(f"**最佳AIC模型**：{best_aic} (AIC = {comparison_df.loc[best_aic, 'AIC']:.2f})")
        
        # 残差分析汇总
        st.markdown("#### 📉 残差分析汇总")
        model_choice = st.selectbox("选择模型查看残差分析", list(models_results.keys()), key='reg_residual_choice')
        if model_choice:
            y_true = models_results[model_choice]['y_true']
            y_pred = models_results[model_choice]['y_pred']
            
            col_res1, col_res2 = st.columns([2, 1])
            with col_res1:
                fig = evaluator.plot_residuals(y_true, y_pred)
                st.pyplot(fig)
            
            with col_res2:
                # Durbin-Watson检验
                residuals = y_true - y_pred
                dw_result = evaluator.durbin_watson_test(residuals)
                st.markdown("**Durbin-Watson检验**")
                st.metric("DW统计量", f"{dw_result['dw_statistic']:.4f}")
                st.info(f"**{dw_result['interpretation']}**")
                
                # 残差统计
                st.markdown("**残差统计**")
                st.metric("均值", f"{np.mean(residuals):.4f}")
                st.metric("标准差", f"{np.std(residuals):.4f}")
    else:
        st.info('💡 请前往"⚔️ 模型竞技场"页面运行回归模型后，评估结果将显示在这里。')
    
    st.markdown("---")
    
    # =======================
    # 2. 分类模型评估
    # =======================
    st.markdown("### 🎯 分类模型评估")
    
    if has_classification:
        clf_eval = st.session_state['classification_evaluation']
        comparison_df = clf_eval['comparison_df']
        evaluator = ModelEvaluator()  # 重新创建evaluator
        class_names = clf_eval['class_names']
        
        # 评估指标对比表
        st.markdown("#### 📊 模型性能对比表")
        st.dataframe(
            comparison_df.style.highlight_max(axis=0),
            use_container_width=True
        )
        
        # 模型选择建议
        st.markdown("#### 💡 模型选择建议")
        best_accuracy = comparison_df['Accuracy'].idxmax()
        best_f1 = comparison_df['F1-Score (Macro)'].idxmax() if 'F1-Score (Macro)' in comparison_df.columns else None
        best_auc = comparison_df['AUC'].idxmax() if 'AUC' in comparison_df.columns and comparison_df['AUC'].notna().any() else None
        
        col_clf1, col_clf2, col_clf3 = st.columns(3)
        with col_clf1:
            st.success(f"**最佳准确率**：{best_accuracy}\n(Accuracy = {comparison_df.loc[best_accuracy, 'Accuracy']:.4f})")
        with col_clf2:
            if best_f1:
                st.success(f"**最佳F1-Score**：{best_f1}\n(F1 = {comparison_df.loc[best_f1, 'F1-Score (Macro)']:.4f})")
        with col_clf3:
            if best_auc:
                st.success(f"**最佳AUC**：{best_auc}\n(AUC = {comparison_df.loc[best_auc, 'AUC']:.4f})")
        
        # 混淆矩阵对比
        st.markdown("#### 🎯 混淆矩阵对比")
        col_cm1, col_cm2, col_cm3 = st.columns(3)
        
        with col_cm1:
            st.markdown("**Logistic Regression**")
            fig, ax = evaluator.plot_confusion_matrix(
                clf_eval['y_true'],
                clf_eval['y_pred_logistic'],
                class_names
            )
            st.pyplot(fig)
        
        with col_cm2:
            st.markdown("**Naive Bayes**")
            fig, ax = evaluator.plot_confusion_matrix(
                clf_eval['y_true'],
                clf_eval['y_pred_nb'],
                class_names
            )
            st.pyplot(fig)
        
        with col_cm3:
            if 'eval_hmm' in clf_eval and clf_eval['eval_hmm'] is not None:
                st.markdown("**HMM**")
                # HMM的混淆矩阵（如果有的话）
                st.info("HMM混淆矩阵需在分类与状态页面查看")
            else:
                st.info("HMM结果未可用")
        
        # ROC曲线对比
        st.markdown("#### 📈 ROC曲线对比")
        try:
            # 绘制多个模型的ROC曲线
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Logistic Regression
            from sklearn.metrics import roc_curve, roc_auc_score
            y_true = clf_eval['y_true']
            y_proba_log = clf_eval['y_proba_logistic']
            
            # 多分类ROC（使用one-vs-rest）
            n_classes = len(class_names)
            if n_classes == 2:
                fpr, tpr, _ = roc_curve(y_true, y_proba_log[:, 1])
                auc = roc_auc_score(y_true, y_proba_log[:, 1])
                ax.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc:.3f})')
            else:
                for i in range(n_classes):
                    y_binary = (y_true == i).astype(int)
                    if len(np.unique(y_binary)) > 1:
                        fpr, tpr, _ = roc_curve(y_binary, y_proba_log[:, i])
                        auc = roc_auc_score(y_binary, y_proba_log[:, i])
                        ax.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc:.3f})')
            
            # Naive Bayes
            y_proba_nb = clf_eval['y_proba_nb']
            if n_classes == 2:
                fpr, tpr, _ = roc_curve(y_true, y_proba_nb[:, 1])
                auc = roc_auc_score(y_true, y_proba_nb[:, 1])
                ax.plot(fpr, tpr, linestyle='--', label=f'Naive Bayes (AUC = {auc:.3f})')
            else:
                for i in range(n_classes):
                    y_binary = (y_true == i).astype(int)
                    if len(np.unique(y_binary)) > 1:
                        fpr, tpr, _ = roc_curve(y_binary, y_proba_nb[:, i])
                        auc = roc_auc_score(y_binary, y_proba_nb[:, i])
                        ax.plot(fpr, tpr, linestyle='--', label=f'{class_names[i]} (NB, AUC = {auc:.3f})')
            
            ax.plot([0, 1], [0, 1], 'k--', label='随机猜测')
            ax.set_xlabel('假正率 (FPR)')
            ax.set_ylabel('真正率 (TPR)')
            ax.set_title('ROC曲线对比')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"ROC曲线绘制失败: {str(e)}")
            # 使用evaluator的方法
            try:
                fig, ax = evaluator.plot_roc_curve(y_true, y_proba_log, class_names)
                st.pyplot(fig)
            except:
                pass
    
    else:
        st.info('💡 请前往"🎯 分类与状态"页面运行分类模型后，评估结果将显示在这里。')
    
    st.markdown("---")
    
    # =======================
    # 3. 综合总结
    # =======================
    st.markdown("### 📋 综合评估总结")
    
    if has_regression and has_classification:
        st.success("""
        **✅ 所有模型评估完成**
        
        **回归模型建议**：
        - 根据R²、AIC/BIC指标选择最佳回归模型
        - 关注残差分析，确保模型假设满足
        
        **分类模型建议**：
        - 根据Accuracy、F1-Score、AUC选择最佳分类模型
        - 关注混淆矩阵，分析各类别的分类性能
        
        **模型选择原则**：
        1. 回归模型：优先考虑R²高、AIC/BIC低的模型
        2. 分类模型：优先考虑Accuracy和F1-Score高的模型
        3. 综合考虑：结合实际应用场景和模型复杂度
        """)
    elif has_regression:
        st.info("回归模型评估已完成，请运行分类模型以获得完整评估。")
    elif has_classification:
        st.info("分类模型评估已完成，请运行回归模型以获得完整评估。")


# ==========================================
# 4. 主程序入口
# ==========================================

def main():
    # 路径设置
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_path = os.path.normpath(os.path.join(current_script_dir, '..', 'Data', 'PRSA_data_2010.1.1-2014.12.31.csv'))

    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3208/3208728.png", width=60)
        st.markdown("### 空气质量监测系统")
        st.markdown("Version 3.0 | Ultimate Edition")
        
        st.markdown("---")
        
        # 漂亮的菜单组件
        if HAS_OPTION_MENU:
            selected = option_menu(
                menu_title=None,
                options=["数据洞察", "归因分析", "⚔️ 模型竞技场", "🎯 分类与状态", "预警中心", "📋 评估中心"],
                icons=["bar-chart-fill", "search", "trophy", "target", "shield-exclamation", "clipboard-data"],
                menu_icon="cast",
                default_index=0,
                styles={
                    "container": {"padding": "0!important", "background-color": "#fafafa"},
                    "icon": {"color": "#2980b9", "font-size": "16px"}, 
                    "nav-link": {"font-size": "14px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                    "nav-link-selected": {"background-color": "#3498db"},
                }
            )
        else:
            selected = st.radio("导航", ["数据洞察", "归因分析", "⚔️ 模型竞技场", "🎯 分类与状态", "预警中心", "📋 评估中心"])
        
        st.markdown("---")
        
        # 数据加载区
        with st.expander("📂 数据管理", expanded=True):
            # 文件上传区域
            st.markdown("#### 📤 上传数据文件")
            uploaded_file = st.file_uploader(
                "选择 CSV 文件",
                type=['csv'],
                help="支持最大200MB的CSV文件，必须包含PM2.5列"
            )
            
            # 文件信息显示和确认
            if uploaded_file is not None:
                # 显示文件基本信息
                file_size_mb = uploaded_file.size / (1024 * 1024)
                
                col_info1, col_info2, col_info3 = st.columns(3)
                with col_info1:
                    st.metric("📄 文件名", uploaded_file.name[:20] + "..." if len(uploaded_file.name) > 20 else uploaded_file.name)
                with col_info2:
                    st.metric("📊 文件大小", f"{file_size_mb:.2f} MB")
                with col_info3:
                    st.metric("📋 文件类型", "CSV")
                
                # 文件验证和预览
                with st.expander("🔍 文件预览与验证", expanded=True):
                    try:
                        # 读取前几行进行预览（使用getvalue()获取副本，不影响原文件指针）
                        import io
                        file_content = uploaded_file.getvalue()
                        preview_df = pd.read_csv(io.StringIO(file_content.decode('utf-8')), nrows=5)
                        
                        st.markdown("**前5行数据预览：**")
                        st.dataframe(preview_df, use_container_width=True)
                        
                        # 检查必需列
                        columns_lower = [col.lower() for col in preview_df.columns]
                        has_pm25 = any('pm2.5' in col or 'pm25' in col or 'pm 2.5' in col for col in columns_lower)
                        has_date = any('date' in col or 'time' in col or 'datetime' in col for col in columns_lower)
                        
                        col_check1, col_check2 = st.columns(2)
                        with col_check1:
                            if has_pm25:
                                st.success("✅ 检测到PM2.5列")
                            else:
                                st.error("❌ 未检测到PM2.5列（必需）")
                        with col_check2:
                            if has_date:
                                st.success("✅ 检测到日期列")
                            else:
                                st.warning("⚠️ 未检测到日期列（时间序列功能可能受限）")
                        
                        # 显示列信息
                        st.markdown(f"**数据列 ({len(preview_df.columns)}个)：**")
                        st.text(", ".join(preview_df.columns.tolist()[:10]))
                        if len(preview_df.columns) > 10:
                            st.text(f"... 还有 {len(preview_df.columns) - 10} 个列")
                        
                    except Exception as e:
                        st.error(f"⚠️ 文件预览失败: {str(e)}")
                        st.info("💡 文件可能不是有效的CSV格式，但仍可尝试加载")
                
                # 确认加载按钮
                st.markdown("---")
                col_btn1, col_btn2 = st.columns([1, 1])
                
                with col_btn1:
                    if st.button("✅ 确认加载此文件", type="primary", use_container_width=True):
                        with st.spinner("正在加载文件..."):
                            try:
                                # 重置文件指针
                                uploaded_file.seek(0)
                                
                                # 显示加载进度（模拟）
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                status_text.text("📥 读取文件...")
                                progress_bar.progress(20)
                                
                                # 加载数据
                                df_uploaded = load_data(uploaded_file)
                                progress_bar.progress(60)
                                
                                if df_uploaded is not None:
                                    status_text.text("✅ 验证数据格式...")
                                    progress_bar.progress(80)
                                    
                                    # 保存数据
                                    st.session_state['data'] = df_uploaded
                                    if 'processed_data' in st.session_state:
                                        del st.session_state['processed_data']
                                    
                                    progress_bar.progress(100)
                                    status_text.text("✅ 加载完成！")
                                    
                                    # 显示成功信息
                                    st.success(f"""
                                    **✅ 文件加载成功！**
                                    
                                    - 📄 文件名: {uploaded_file.name}
                                    - 📊 数据量: {len(df_uploaded):,} 条记录
                                    - 📋 列数: {len(df_uploaded.columns)} 个
                                    - 🌫️ PM2.5范围: {df_uploaded['PM2.5'].min():.1f} ~ {df_uploaded['PM2.5'].max():.1f} μg/m³
                                    """)
                                    
                                    # 延迟后刷新页面
                                    import time
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    progress_bar.progress(0)
                                    status_text.empty()
                                    st.error("""
                                    **❌ 文件加载失败**
                                    
                                    可能的原因：
                                    - 文件不包含PM2.5列（必需）
                                    - 文件格式不正确
                                    - 文件编码问题
                                    
                                    💡 请检查文件格式，确保包含PM2.5列
                                    """)
                            except Exception as e:
                                progress_bar.progress(0)
                                status_text.empty()
                                st.error(f"""
                                **❌ 文件加载出错**
                                
                                错误信息: {str(e)}
                                
                                💡 请检查：
                                - 文件是否为有效的CSV格式
                                - 文件编码是否为UTF-8
                                - 文件是否损坏
                                """)
                                import traceback
                                with st.expander("🔍 查看详细错误信息"):
                                    st.code(traceback.format_exc())
                
                with col_btn2:
                    if st.button("❌ 取消", use_container_width=True):
                        # 清除上传的文件（通过刷新）
                        st.session_state.pop('uploaded_file', None)
                        st.rerun()
            
            else:
                # 未上传文件时的提示
                st.info("💡 请上传CSV文件，或使用下方的测试数据")
            
            st.markdown("---")
            st.markdown("### 🧹 预处理设置")
            st.caption("💡 修改设置后会自动重新处理数据")
            
            # 初始化预处理设置（如果不存在）
            if 'preprocessing_settings' not in st.session_state:
                st.session_state['preprocessing_settings'] = {
                    'missing_method': 'interpolation',
                    'outlier_method': '3sigma',
                    'do_log': False
                }
            
            # 预处理设置选择
            missing_method = st.selectbox(
                "缺失值处理",
                ["interpolation", "drop"],
                index=0 if st.session_state['preprocessing_settings']['missing_method'] == 'interpolation' else 1,
                help="interpolation: 线性插值填补缺失值 | drop: 删除包含缺失值的行"
            )
            outlier_method = st.selectbox(
                "异常值处理",
                ["3sigma", "iqr", "none"],
                index=["3sigma", "iqr", "none"].index(st.session_state['preprocessing_settings']['outlier_method']),
                help="3sigma: 3倍标准差原则 | iqr: 四分位距方法 | none: 不处理异常值"
            )
            do_log = st.checkbox(
                "对 PM2.5 做 Log 变换（用于检验/建模对比）",
                value=st.session_state['preprocessing_settings']['do_log'],
                help="对PM2.5进行对数变换，使其更接近正态分布"
            )
            
            # 检查设置是否改变
            settings_changed = (
                missing_method != st.session_state['preprocessing_settings']['missing_method'] or
                outlier_method != st.session_state['preprocessing_settings']['outlier_method'] or
                do_log != st.session_state['preprocessing_settings']['do_log']
            )
            
            if settings_changed:
                # 更新设置
                st.session_state['preprocessing_settings'] = {
                    'missing_method': missing_method,
                    'outlier_method': outlier_method,
                    'do_log': do_log
                }
                # 清除已处理的数据缓存，强制重新处理
                if 'processed_data' in st.session_state:
                    del st.session_state['processed_data']
                st.rerun()
            
            st.markdown("---")
            st.markdown("### 🧪 或使用测试数据")
            st.caption("快速加载项目自带的UCI Beijing PM2.5数据集（2010-2014年）")
            if st.button("🔄 加载测试数据", use_container_width=True):
                if os.path.exists(default_data_path):
                    st.session_state['data'] = load_data(default_data_path)
                    if 'processed_data' in st.session_state: 
                        del st.session_state['processed_data']
                    st.success("✅ 测试数据加载成功")
                    st.rerun()
                else:
                    # 尝试其他可能的文件名
                    alternative_paths = [
                        os.path.normpath(os.path.join(current_script_dir, '..', 'Data', 'PRSA_data.csv')),
                        os.path.normpath(os.path.join(current_script_dir, '..', 'Data', 'beijing+pm2+5+data', 'PRSA_data.csv')),
                    ]
                    found = False
                    for alt_path in alternative_paths:
                        if os.path.exists(alt_path):
                            st.session_state['data'] = load_data(alt_path)
                            if 'processed_data' in st.session_state: 
                                del st.session_state['processed_data']
                            st.success(f"✅ 找到数据文件: {alt_path}")
                            st.rerun()
                            found = True
                            break
                    if not found:
                        st.error(f"❌ 测试文件未找到。请检查以下路径：\n- {default_data_path}\n- {alternative_paths[0]}")
                        st.info("💡 提示：您也可以使用上方的文件上传功能上传CSV文件")
                      
        if 'data' in st.session_state:
            st.success(f"📊 已加载 {len(st.session_state['data'])} 条数据")

    # 主逻辑路由
    if 'data' in st.session_state:
        df = st.session_state['data'].copy()
        
        # 从session_state获取预处理设置
        if 'preprocessing_settings' not in st.session_state:
            # 如果设置不存在，使用默认值
            preprocessing_settings = {
                'missing_method': 'interpolation',
                'outlier_method': '3sigma',
                'do_log': False
            }
        else:
            preprocessing_settings = st.session_state['preprocessing_settings']
        
        missing_method = preprocessing_settings['missing_method']
        outlier_method = preprocessing_settings['outlier_method']
        do_log = preprocessing_settings['do_log']
        
        if 'processed_data' not in st.session_state:
            with st.spinner("正在进行智能清洗与预处理..."):
                df_processed = preprocess_data(
                    df,
                    missing_method=missing_method,
                    outlier_method=outlier_method,
                    do_log=do_log
                )

                st.session_state['processed_data'] = df_processed
        else:
            df_processed = st.session_state['processed_data']
        
        # 页面跳转
        if selected == "数据洞察":
            page_data_insight(df_processed)
        elif selected == "归因分析":
            page_attribution_analysis(df_processed)
        elif selected == "⚔️ 模型竞技场" or selected == "模型竞技场":
            page_model_arena(df_processed)
        elif selected == "🎯 分类与状态" or selected == "分类与状态":
            page_classification(df_processed)
        elif selected == "预警中心":
            page_warning_center(df_processed)
        elif selected == "📋 评估中心" or selected == "评估中心":
            page_evaluation_center(df_processed)
            
    else:
        # 欢迎页（空状态）
        st.markdown("""
        <div style="display: flex; justify-content: center; align-items: center; height: 60vh; flex-direction: column;">
            <h2 style="color: #ccc;">👋 欢迎使用系统</h2>
            <p style="color: #999;">请在左侧侧边栏加载数据以开始分析</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()