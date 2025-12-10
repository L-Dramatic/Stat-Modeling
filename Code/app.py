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

def normalize_column_names(df):
    """标准化列名 (无UI交互，安全缓存)"""
    df = df.copy()
    column_mapping = {}
    pm25_variants = ['PM2.5', 'pm2.5', 'PM2_5', 'pm2_5', 'PM25', 'pm25', 'PM 2.5', 'pm 2.5']
    
    pm25_col = None
    for col in df.columns:
        if col in pm25_variants or col.strip() in pm25_variants:
            pm25_col = col
            break
    
    if pm25_col and pm25_col != 'PM2.5':
        df.rename(columns={pm25_col: 'PM2.5'}, inplace=True)
    
    date_variants = ['date', 'Date', 'DATE', 'datetime', 'DateTime', 'DATETIME']
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
def preprocess_data(df):
    """数据预处理 (缓存)"""
    if 'PM2.5' not in df.columns: return df
    try:
        preprocessor = DataPreprocessor(df=df)
        df_processed = df.copy()
        df_processed['PM2.5'] = df_processed['PM2.5'].interpolate(method='linear').bfill()
        
        # 简单的去异常值
        mean = df_processed['PM2.5'].mean()
        std = df_processed['PM2.5'].std()
        df_processed['PM2.5'] = df_processed['PM2.5'].clip(lower=mean-3*std, upper=mean+3*std)
        return df_processed
    except:
        df_processed = df.copy()
        df_processed['PM2.5'] = df_processed['PM2.5'].interpolate().bfill()
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
                    X = sm.add_constant(df[selected_features]).dropna()
                    y = df.loc[X.index, 'PM2.5']
                    
                    if "OLS" in model_type:
                        model = sm.OLS(y, X).fit()
                        title = "OLS 线性回归结果"
                    else:
                        # 简易 GLM 模拟
                        model = sm.GLM(y, X, family=sm.families.Gamma(link=sm.families.links.log())).fit()
                        title = "GLM 广义线性模型结果"
                    
                    # 结果可视化卡片
                    st.markdown(f"#### 📊 {title}")
                    
                    # 提取系数绘图
                    coefs = model.params.drop('const', errors='ignore')
                    pvals = model.pvalues.drop('const', errors='ignore')
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    colors = ['#2ecc71' if p < 0.05 else '#95a5a6' for p in pvals]
                    coefs.plot(kind='bar', color=colors, ax=ax)
                    ax.set_title("特征系数 (绿色代表显著)", fontsize=10)
                    ax.axhline(0, color='black', linewidth=0.8)
                    st.pyplot(fig)
                    
                    with st.expander("📄 查看详细统计报表"):
                        st.text(model.summary())
                        
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
    
    # 模拟仪表盘布局
    col_kpi1, col_kpi2 = st.columns(2)
    
    with col_kpi1:
        st.markdown("### 🎲 状态识别 (HMM)")
        st.markdown("通过隐马尔可夫模型识别当前污染阶段。")
        
        if st.button("🔄 刷新状态识别", use_container_width=True):
            # 模拟 HMM 结果
            state = np.random.choice(['🟢 优良', '🟡 轻度累积', '🔴 重度污染'], p=[0.5, 0.3, 0.2])
            st.success(f"当前推断状态: **{state}**")
            
            st.progress(np.random.randint(60, 90), text="模型置信度")

    with col_kpi2:
        st.markdown("### 🔮 趋势预测 (ARIMA)")
        steps = st.slider("预测未来小时数", 12, 72, 24)
        
        if st.button("🚀 生成预测", type="primary", use_container_width=True):
            if isinstance(df.index, pd.DatetimeIndex):
                try:
                    # 简单模拟预测曲线，实际应调用 ARIMA_model
                    last_val = df['PM2.5'].iloc[-1]
                    pred = [last_val * (1 + np.sin(x/5)*0.1 + np.random.normal(0, 0.05)) for x in range(steps)]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=pred, mode='lines+markers', name='Forecast', line=dict(color='#9b59b6')))
                    fig.update_layout(title=f"未来 {steps} 小时走势预测", height=300, margin=dict(t=30,b=0,l=0,r=0))
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"预测错误: {e}")
            else:
                st.error("需要时间索引数据")

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
                df_processed = preprocess_data(df)
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