import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
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

# 添加Code目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入你的模块
try:
    from data_preprocessing import DataPreprocessor
    from statistical_inference import StatisticalInference
    from glm_model import GLMModel
    from arima_model import ARIMAModel
    from hmm_model import HMMModel
except ImportError:
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

    steps = st.slider("预测未来小时数", 12, 72, 24)
    run_arima = st.button("📈 生成 ARIMA 预测", type="primary", use_container_width=True)

    if run_arima:
        with st.spinner("ARIMA 拟合与预测中..."):
            series = df["PM2.5"].dropna()

            arima = ARIMAModel()

            # 平稳性检验
            stat_res = arima.check_stationarity(series)
            st.write("ADF 检验结果：", stat_res)

            # 拟合（自动选参）
            arima.fit(series, auto_select=True)

            # 预测
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
        st.markdown("Version 2.0 | Pro Edition")
        
        st.markdown("---")
        
        # 漂亮的菜单组件
        if HAS_OPTION_MENU:
            selected = option_menu(
                menu_title=None,
                options=["数据洞察", "归因分析", "预警中心"],
                icons=["bar-chart-fill", "search", "shield-exclamation"],
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
            selected = st.radio("导航", ["数据洞察", "归因分析", "预警中心"])
        
        st.markdown("---")
        
        # 数据加载区
        with st.expander("📂 数据管理", expanded=True):
            uploaded_file = st.file_uploader("上传 CSV", type=['csv'])
            st.markdown("### 🧹 预处理设置")
            missing_method = st.selectbox("缺失值处理", ["interpolation", "drop"], index=0)
            outlier_method = st.selectbox("异常值处理", ["3sigma", "iqr", "none"], index=0)
            do_log = st.checkbox("对 PM2.5 做 Log 变换（用于检验/建模对比）", value=False)    
            if st.button("🔄 加载测试数据"):
                if os.path.exists(default_data_path):
                    st.session_state['data'] = load_data(default_data_path)
                    if 'processed_data' in st.session_state: del st.session_state['processed_data']
                    st.rerun()
                else:
                    st.error("测试文件未找到")              
        if 'data' in st.session_state:
            st.success(f"已加载 {len(st.session_state['data'])} 条数据")

    # 主逻辑路由
    if 'data' in st.session_state:
        df = st.session_state['data'].copy()
        
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
        elif selected == "预警中心":
            page_warning_center(df_processed)
            
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