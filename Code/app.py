"""
Streamlit主应用
基于时间序列与HMM的城市空气质量监测及归因预警系统

页面A：数据洞察 (Data Insight)
页面B：归因分析 (Attribution Analysis)
页面C：预警中心 (Warning Center)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# 设置matplotlib和seaborn的全局样式
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
sns.set_style("whitegrid", {
    'axes.grid': True,
    'axes.edgecolor': '.8',
    'axes.linewidth': 1.5,
    'grid.color': '.9',
    'grid.linewidth': 1,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# 添加Code目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import DataPreprocessor
from statistical_inference import StatisticalInference
from glm_model import GLMModel
from arima_model import ARIMAModel
from hmm_model import HMMModel

# 页面配置
st.set_page_config(
    page_title="空气质量监测预警系统",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 精美的自定义CSS - 专业精致版
st.markdown("""
<style>
    /* 全局样式 - 优雅的浅色背景 */
    .stApp {
        background: #f8f9fa;
    }
    
    /* 主容器 */
    .main .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }
    
    /* 主标题 - 简洁优雅 */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1a1a1a;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 0;
        letter-spacing: -0.5px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* 副标题 - 优雅的深色 */
    h1, h2, h3 {
        color: #1a1a1a;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    h2 {
        color: #2c3e50;
        font-size: 1.8rem;
        font-weight: 600;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #e9ecef;
        margin-bottom: 1.5rem;
    }
    
    h3 {
        color: #495057;
        font-size: 1.4rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* 指标卡片 - 简洁白色卡片 */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        color: #1a1a1a;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid #e9ecef;
        transition: all 0.2s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
        transform: translateY(-2px);
    }
    
    /* 信息框 - 柔和的蓝色 */
    .stInfo {
        background: #f0f7ff;
        border-left: 4px solid #0066cc;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    /* 成功框 - 柔和的绿色 */
    .stSuccess {
        background: #f0fdf4;
        border-left: 4px solid #10b981;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    /* 警告框 - 柔和的黄色 */
    .stWarning {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    /* 错误框 - 柔和的红色 */
    .stError {
        background: #fef2f2;
        border-left: 4px solid #ef4444;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    /* 按钮样式 - 专业的蓝色 */
    .stButton > button {
        background: #0066cc;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.625rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(0, 102, 204, 0.2);
        font-size: 0.95rem;
    }
    
    .stButton > button:hover {
        background: #0052a3;
        box-shadow: 0 4px 8px rgba(0, 102, 204, 0.3);
        transform: translateY(-1px);
    }
    
    /* 侧边栏样式 - 深色专业 */
    [data-testid="stSidebar"] {
        background: #1e293b;
    }
    
    [data-testid="stSidebar"] .css-1lcbmhc {
        color: #f1f5f9;
    }
    
    /* 选择框样式 */
    .stSelectbox label, .stMultiselect label, .stRadio label {
        font-weight: 500;
        color: #495057;
        font-size: 0.95rem;
    }
    
    /* 数据框样式 - 简洁白色 */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        border: 1px solid #e9ecef;
    }
    
    /* 标签页样式 - 简洁设计 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 2px solid #e9ecef;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.25rem;
        font-weight: 500;
        transition: all 0.2s ease;
        color: #6c757d;
        border-bottom: 2px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: transparent;
        color: #0066cc;
        border-bottom: 2px solid #0066cc;
        font-weight: 600;
    }
    
    /* 指标数字样式 - 深色专业 */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a1a1a;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    [data-testid="stMetricLabel"] {
        color: #6c757d;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* 分隔线 - 简洁 */
    hr {
        border: none;
        height: 1px;
        background: #e9ecef;
        margin: 2rem 0;
    }
    
    /* 代码块样式 */
    .stCodeBlock {
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        border: 1px solid #e9ecef;
    }
    
    /* 展开框样式 */
    .streamlit-expanderHeader {
        background: #f8f9fa;
        border-radius: 8px;
        font-weight: 500;
        padding: 0.75rem 1rem;
        border: 1px solid #e9ecef;
    }
    
    /* 滑块样式 */
    .stSlider {
        padding: 1rem 0;
    }
    
    /* 主内容区域卡片效果 */
    .element-container {
        margin-bottom: 1.5rem;
    }
    
    /* 图表容器 */
    [data-testid="stPlotlyChart"] {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        border: 1px solid #e9ecef;
    }
    
    /* 滚动条样式 - 简洁 */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f3f5;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #adb5bd;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #868e96;
    }
    
    /* 侧边栏文字颜色 */
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #f1f5f9;
    }
    
    /* 输入框样式 */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        border: 1px solid #dee2e6;
        border-radius: 6px;
    }
    
    /* 整体间距优化 */
    .main .block-container {
        padding-left: 3rem;
        padding-right: 3rem;
    }
</style>
""", unsafe_allow_html=True)


def normalize_column_names(df):
    """
    标准化列名，自动识别PM2.5列（不区分大小写）
    
    Parameters:
    -----------
    df : pd.DataFrame
        原始数据框
    
    Returns:
    --------
    pd.DataFrame : 列名标准化后的数据框
    dict : 列名映射字典
    """
    df = df.copy()
    column_mapping = {}
    
    # 查找PM2.5列（可能的变体）
    pm25_variants = ['PM2.5', 'pm2.5', 'PM2_5', 'pm2_5', 'PM25', 'pm25', 
                     'PM 2.5', 'pm 2.5', 'PM_2.5', 'pm_2.5']
    
    pm25_col = None
    for col in df.columns:
        if col in pm25_variants or col.strip() in pm25_variants:
            pm25_col = col
            break
    
    # 如果找到PM2.5列，标准化为'PM2.5'
    if pm25_col and pm25_col != 'PM2.5':
        df.rename(columns={pm25_col: 'PM2.5'}, inplace=True)
        column_mapping[pm25_col] = 'PM2.5'
        st.info(f"📝 检测到PM2.5列: '{pm25_col}' → 'PM2.5'")
    
    # 标准化日期列
    date_variants = ['date', 'Date', 'DATE', 'datetime', 'DateTime', 'DATETIME',
                     'time', 'Time', 'TIME', 'timestamp', 'Timestamp', 'TIMESTAMP']
    date_col = None
    for col in df.columns:
        if col in date_variants or col.strip() in date_variants:
            date_col = col
            break
    
    if date_col and date_col != 'date':
        df.rename(columns={date_col: 'date'}, inplace=True)
        column_mapping[date_col] = 'date'
    
    return df, column_mapping


@st.cache_data
def load_data(file_path):
    """加载数据（缓存）"""
    try:
        if isinstance(file_path, str):
            df = pd.read_csv(file_path)
        else:
            # 如果是上传的文件对象
            df = pd.read_csv(file_path)
        
        # 标准化列名
        df, mapping = normalize_column_names(df)
        
        # 检查必需的列
        if 'PM2.5' not in df.columns:
            # 显示所有列名供用户参考
            available_cols = ', '.join(df.columns.tolist()[:10])
            if len(df.columns) > 10:
                available_cols += f", ... (共{len(df.columns)}列)"
            st.error(f"❌ 数据中未找到PM2.5列！\n\n可用列名: {available_cols}")
            st.info("💡 提示：PM2.5列名可以是：PM2.5, pm2.5, PM2_5, pm2_5, PM25, pm25等")
            return None
        
        return df
    except Exception as e:
        st.error(f"数据加载失败: {str(e)}")
        return None


@st.cache_data
def preprocess_data(df):
    """预处理数据（缓存）"""
    # 确保列名已标准化
    if 'PM2.5' not in df.columns:
        df, _ = normalize_column_names(df)
    
    if 'PM2.5' not in df.columns:
        st.error("❌ 预处理失败：数据中缺少PM2.5列")
        return df
    
    preprocessor = DataPreprocessor(df=df)
    preprocessor.handle_missing_values(method='interpolation')
    preprocessor.remove_outliers(column='PM2.5', method='3sigma')
    return preprocessor.get_processed_data()


def page_data_insight(df):
    """页面A：数据洞察"""
    # 精美的页面标题 - 简洁专业
    st.markdown("""
    <div style="text-align: left; padding: 1.5rem 0; margin-bottom: 2rem; border-bottom: 2px solid #e9ecef;">
        <h1 style="color: #1a1a1a; font-size: 2rem; font-weight: 700; margin: 0 0 0.5rem 0;">
            📊 数据洞察
        </h1>
        <p style="color: #6c757d; font-size: 0.95rem; margin: 0; font-weight: 400;">
            Data Insight & Exploratory Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 检查PM2.5列是否存在
    if 'PM2.5' not in df.columns:
        st.error("❌ 数据中缺少PM2.5列！")
        st.info(f"**当前数据列名：** {', '.join(df.columns.tolist())}")
        st.warning("💡 请确保数据包含PM2.5列（列名可以是：PM2.5, pm2.5, PM2_5等）")
        return
    
    # 数据概览
    st.subheader("数据概览")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总记录数", len(df))
    with col2:
        pm25_mean = df['PM2.5'].mean() if not df['PM2.5'].isna().all() else 0
        st.metric("PM2.5均值", f"{pm25_mean:.2f} μg/m³")
    with col3:
        pm25_max = df['PM2.5'].max() if not df['PM2.5'].isna().all() else 0
        st.metric("PM2.5最大值", f"{pm25_max:.2f} μg/m³")
    with col4:
        st.metric("缺失值", df['PM2.5'].isna().sum())
    
    # 1. PM2.5历史趋势图
    st.markdown("### 📉 PM2.5历史趋势")
    if 'date' in df.index.names or isinstance(df.index, pd.DatetimeIndex):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['PM2.5'],
            mode='lines',
            name='PM2.5',
            line=dict(
                color='#667eea',
                width=2,
                shape='spline',
                smoothing=1.3
            ),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.1)',
            hovertemplate='<b>日期</b>: %{x}<br><b>PM2.5</b>: %{y:.2f} μg/m³<extra></extra>'
        ))
        fig.update_layout(
            title=dict(
                text="PM2.5时间序列",
                font=dict(size=20, color='#2c3e50', family='Arial Black')
            ),
            xaxis=dict(
                title="日期",
                titlefont=dict(size=14, color='#2c3e50'),
                gridcolor='rgba(128, 128, 128, 0.2)',
                showgrid=True
            ),
            yaxis=dict(
                title="PM2.5 (μg/m³)",
                titlefont=dict(size=14, color='#2c3e50'),
                gridcolor='rgba(128, 128, 128, 0.2)',
                showgrid=True
            ),
            height=450,
            plot_bgcolor='rgba(255, 255, 255, 0.9)',
            paper_bgcolor='rgba(255, 255, 255, 0.9)',
            hovermode='x unified',
            font=dict(family="Arial", size=12)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("数据缺少日期索引，无法绘制时间序列图")
    
    # 2. 相关分析热力图
    st.markdown("### 🔗 相关分析（Correlation Analysis）")
    st.info("💡 分析PM2.5与气象因子的相关系数，检测多重共线性")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'PM2.5' in numeric_cols and len(numeric_cols) > 1:
        # 选择与PM2.5相关的数值列
        corr_cols = [col for col in numeric_cols if col not in ['No', 'year', 'month', 'day', 'hour']][:10]
        corr_data = df[corr_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
        sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0,
                   square=True, linewidths=1.5, linecolor='white',
                   cbar_kws={"shrink": 0.8, "label": "相关系数"},
                   annot_kws={"size": 10, "weight": "bold"},
                   ax=ax, vmin=-1, vmax=1)
        ax.set_title('PM2.5与气象因子相关系数热力图', fontsize=16, fontweight='bold', pad=25)
        plt.xticks(rotation=45, ha='right', fontsize=11, fontweight='bold')
        plt.yticks(rotation=0, fontsize=11, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        
        # 显示与PM2.5相关性最强的因子
        if 'PM2.5' in corr_data.columns:
            pm25_corr = corr_data['PM2.5'].drop('PM2.5').abs().sort_values(ascending=False)
            st.write("**与PM2.5相关性最强的因子（按绝对值排序）**")
            corr_df = pd.DataFrame({
                '因子': pm25_corr.index,
                '相关系数': [corr_data.loc[idx, 'PM2.5'] for idx in pm25_corr.index],
                '绝对值': pm25_corr.values
            })
            st.dataframe(corr_df.head(10), use_container_width=True)
    
    # 3. 正态性检验（直方图 vs 拟合曲线 + QQ图）
    st.markdown("### 📊 正态性检验与分布拟合")
    
    preprocessor = DataPreprocessor(df=df)
    
    # 创建标签页
    tab1, tab2, tab3 = st.tabs(["原始数据分布", "Log变换后", "分布拟合结果"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # 精美的直方图
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
            n, bins, patches = ax.hist(df['PM2.5'].dropna(), bins=50, density=True, 
                                       alpha=0.8, edgecolor='white', linewidth=1.5)
            
            # 渐变色
            colors = plt.cm.viridis(np.linspace(0, 1, len(patches)))
            for patch, color in zip(patches, colors):
                patch.set_facecolor(color)
            
            # 拟合正态分布
            mu, sigma = df['PM2.5'].mean(), df['PM2.5'].std()
            x = np.linspace(df['PM2.5'].min(), df['PM2.5'].max(), 100)
            ax.plot(x, np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi)),
                    'r-', linewidth=3, label='正态分布拟合', alpha=0.8)
            ax.set_xlabel('PM2.5 (μg/m³)', fontsize=12, fontweight='bold')
            ax.set_ylabel('密度', fontsize=12, fontweight='bold')
            ax.set_title('PM2.5分布直方图 vs 正态分布拟合', fontsize=14, fontweight='bold', pad=20)
            ax.legend(fontsize=11, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # QQ图
            from scipy import stats as scipy_stats
            fig, ax = plt.subplots(figsize=(8, 5))
            scipy_stats.probplot(df['PM2.5'].dropna(), dist="norm", plot=ax)
            ax.set_title('Q-Q图（检验正态性）')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # 正态性检验
            normality_test = preprocessor.test_normality(df['PM2.5'].dropna(), test_type='normaltest')
            st.write(f"**{normality_test['test_name']}检验**")
            st.metric("统计量", f"{normality_test['statistic']:.4f}")
            st.metric("P值", f"{normality_test['p_value']:.4f}")
            if normality_test['is_normal']:
                st.success(f"✅ {normality_test['interpretation']}")
            else:
                st.warning(f"⚠️ {normality_test['interpretation']}")
    
    with tab2:
        st.info("💡 Log变换可以使右偏数据近似正态分布，这是时间序列和回归模型的前提")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Log变换后的直方图
            log_pm25 = preprocessor.log_transform(column='PM2.5')
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(log_pm25.dropna(), bins=50, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
            
            # 拟合正态分布
            mu_log, sigma_log = log_pm25.mean(), log_pm25.std()
            x_log = np.linspace(log_pm25.min(), log_pm25.max(), 100)
            ax.plot(x_log, np.exp(-0.5 * ((x_log - mu_log) / sigma_log) ** 2) / (sigma_log * np.sqrt(2 * np.pi)),
                    'r-', linewidth=2, label='正态分布拟合')
            ax.set_xlabel('Log(PM2.5)')
            ax.set_ylabel('密度')
            ax.set_title('Log变换后的PM2.5分布')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            # Log变换后的QQ图
            fig, ax = plt.subplots(figsize=(8, 5))
            scipy_stats.probplot(log_pm25.dropna(), dist="norm", plot=ax)
            ax.set_title('Log变换后的Q-Q图')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Log变换后的正态性检验
            normality_test_log = preprocessor.test_normality(log_pm25.dropna(), test_type='normaltest')
            st.write(f"**{normality_test_log['test_name']}检验（Log变换后）**")
            st.metric("统计量", f"{normality_test_log['statistic']:.4f}")
            st.metric("P值", f"{normality_test_log['p_value']:.4f}")
            if normality_test_log['is_normal']:
                st.success(f"✅ Log变换后{normality_test_log['interpretation']}")
            else:
                st.info(f"ℹ️ Log变换后{normality_test_log['interpretation']}（但仍比原始数据更接近正态）")
    
    with tab3:
        # 分布拟合结果
        dist_results = preprocessor.fit_distribution(column='PM2.5')
        
        st.write("**分布拟合结果（Kolmogorov-Smirnov检验）**")
        results_df = []
        for dist_name, result in dist_results.items():
            if dist_name != 'best_fit' and 'error' not in result:
                results_df.append({
                    '分布': dist_name,
                    'KS统计量': f"{result['ks_statistic']:.4f}",
                    'P值': f"{result['p_value']:.4f}",
                    'AIC': f"{result['aic']:.2f}"
                })
        
        if results_df:
            results_df = pd.DataFrame(results_df)
            st.dataframe(results_df, use_container_width=True)
            
            if 'best_fit' in dist_results:
                st.success(f"✅ 最佳拟合分布: **{dist_results['best_fit']}**")
                st.info("💡 基于分布拟合结果，我们选择GLM的Gamma分布族进行建模（而非普通线性回归）")
    
    # 4. 假设检验：工作日vs周末（T检验）
    st.markdown("### 🧪 假设检验：工作日 vs 周末（T检验）")
    st.info("💡 检验人类活动（工作日vs周末）对空气质量的影响")
    
    # 创建工作日/周末分组
    if isinstance(df.index, pd.DatetimeIndex) or 'date' in df.index.names:
        df_with_weekend = df.copy()
        if 'is_weekend' not in df_with_weekend.columns:
            df_with_weekend['is_weekend'] = df_with_weekend.index.weekday >= 5
            df_with_weekend['day_type'] = df_with_weekend['is_weekend'].map({True: '周末', False: '工作日'})
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 箱线图
            fig, ax = plt.subplots(figsize=(8, 5))
            df_with_weekend.boxplot(column='PM2.5', by='day_type', ax=ax)
            ax.set_xlabel('')
            ax.set_ylabel('PM2.5 (μg/m³)')
            ax.set_title('工作日 vs 周末的PM2.5分布')
            plt.suptitle('')  # 移除默认标题
            st.pyplot(fig)
        
        with col2:
            # T检验结果
            inference = StatisticalInference(df_with_weekend)
            ttest_result = inference.t_test(column='PM2.5', group_column='is_weekend')
            
            st.write("**独立样本T检验结果**")
            st.metric("T统计量", f"{ttest_result['t_statistic']:.4f}")
            st.metric("P值", f"{ttest_result['p_value']:.4f}")
            
            if ttest_result['significant']:
                st.success("✅ 工作日和周末的PM2.5存在显著差异")
                st.info("💡 说明人类活动对空气质量有显著影响")
            else:
                st.warning("⚠️ 工作日和周末的PM2.5无显著差异")
            
            st.write("**各组统计量**")
            for group, count in ttest_result['groups'].items():
                group_name = '周末' if group else '工作日'
                mean_val = ttest_result['group1_mean'] if group else ttest_result['group2_mean']
                st.write(f"- {group_name}: 均值={mean_val:.2f}, 样本数={count}")
    else:
        st.warning("数据缺少日期信息，无法进行工作日/周末分组")
    
    # 5. 风向对污染影响的箱线图（ANOVA结果）
    
    # 5. 风向对污染影响的箱线图（ANOVA结果）
    st.markdown("### 🌬️ 方差分析：风向对PM2.5的影响（ANOVA）")
    
    if 'cbwd' in df.columns:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 箱线图
            fig, ax = plt.subplots(figsize=(10, 6))
            df.boxplot(column='PM2.5', by='cbwd', ax=ax)
            ax.set_xlabel('风向')
            ax.set_ylabel('PM2.5 (μg/m³)')
            ax.set_title('不同风向下的PM2.5分布')
            plt.suptitle('')  # 移除默认标题
            st.pyplot(fig)
        
        with col2:
            # ANOVA结果
            inference = StatisticalInference(df)
            anova_result = inference.anova_test(column='PM2.5', group_column='cbwd')
            
            st.write("**ANOVA检验结果**")
            st.metric("F统计量", f"{anova_result['f_statistic']:.4f}")
            st.metric("P值", f"{anova_result['p_value']:.4f}")
            
            if anova_result['significant']:
                st.success("✅ 不同风向下的PM2.5存在显著差异")
            else:
                st.warning("⚠️ 不同风向下的PM2.5无显著差异")
            
            st.write("**各组统计量**")
            for group, stats in anova_result['groups'].items():
                st.write(f"- {group}: 均值={stats['mean']:.2f}, 样本数={stats['count']}")
    else:
        st.warning("数据中缺少'cbwd'（风向）列")


def page_attribution_analysis(df):
    """页面B：归因分析"""
    # 精美的页面标题 - 简洁专业
    st.markdown("""
    <div style="text-align: left; padding: 1.5rem 0; margin-bottom: 2rem; border-bottom: 2px solid #e9ecef;">
        <h1 style="color: #1a1a1a; font-size: 2rem; font-weight: 700; margin: 0 0 0.5rem 0;">
            🔍 归因分析
        </h1>
        <p style="color: #6c757d; font-size: 0.95rem; margin: 0; font-weight: 400;">
            Attribution Analysis & Regression Modeling
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 检查PM2.5列是否存在
    if 'PM2.5' not in df.columns:
        st.error("❌ 数据中缺少PM2.5列！")
        st.info(f"**当前数据列名：** {', '.join(df.columns.tolist())}")
        return
    
    st.info("💡 使用回归模型分析气象因子对PM2.5的影响。先建立OLS baseline，再使用GLM（Gamma分布族）进行优化")
    
    # 特征选择
    st.markdown("### 🎯 特征选择")
    
    # 获取数值特征
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'PM2.5' in numeric_cols:
        numeric_cols.remove('PM2.5')
    
    # 排除可能的ID列
    feature_options = [col for col in numeric_cols if col not in ['No', 'year', 'month', 'day', 'hour']]
    
    selected_features = st.multiselect(
        "选择自变量（气象因子）",
        options=feature_options,
        default=feature_options[:5] if len(feature_options) >= 5 else feature_options
    )
    
    if len(selected_features) == 0:
        st.warning("请至少选择一个特征")
        return
    
    # 多重共线性检测
    st.markdown("### 🔍 多重共线性检测")
    inference = StatisticalInference(df)
    multicollinearity = inference.detect_multicollinearity(threshold=0.8)
    
    if multicollinearity:
        st.warning("⚠️ 检测到高相关变量对（相关系数 > 0.8）：")
        for pair in multicollinearity:
            st.write(f"- {pair['var1']} 与 {pair['var2']}: {pair['correlation']:.4f}")
        st.info("💡 建议从高相关变量对中只保留一个变量")
    else:
        st.success("✅ 未检测到严重的多重共线性问题")
    
    # 准备数据
    X = df[selected_features].copy()
    y = df['PM2.5'].copy()
    
    # 处理缺失值
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    if len(X) == 0:
        st.error("数据准备失败，请检查特征选择")
        return
    
    # 模型选择
    model_type = st.radio(
        "选择模型类型",
        ["OLS（普通线性回归，Baseline）", "GLM（广义线性模型，Gamma分布族）"],
        horizontal=True
    )
    
    # 拟合模型
    if st.button("运行模型", type="primary"):
        with st.spinner("正在拟合模型..."):
            if "OLS" in model_type:
                # OLS模型
                import statsmodels.api as sm
                X_with_const = sm.add_constant(X)
                ols_model = sm.OLS(y, X_with_const).fit()
                st.session_state['ols_model'] = ols_model
                st.session_state['model_type'] = 'OLS'
            else:
                # GLM模型
                glm = GLMModel(family='gamma', link='log')
                glm.fit(X, y)
                st.session_state['glm_model'] = glm
                st.session_state['model_type'] = 'GLM'
            
            st.session_state['model_X'] = X
            st.session_state['model_y'] = y
    
    # 显示模型结果
    if 'model_type' in st.session_state:
        model_type_used = st.session_state['model_type']
        
        if model_type_used == 'OLS' and 'ols_model' in st.session_state:
            ols_model = st.session_state['ols_model']
            
            st.subheader("OLS模型统计摘要（Baseline）")
            st.text(str(ols_model.summary()))
            
            # 显著特征
            st.subheader("显著特征（P < 0.05）")
            ols_summary = pd.DataFrame({
                '系数': ols_model.params,
                '标准误': ols_model.bse,
                'T值': ols_model.tvalues,
                'P值': ols_model.pvalues,
                '置信区间下界': ols_model.conf_int()[0],
                '置信区间上界': ols_model.conf_int()[1]
            })
            significant_ols = ols_summary[ols_summary['P值'] < 0.05].sort_values('P值')
            
            if len(significant_ols) > 0:
                st.dataframe(significant_ols, use_container_width=True)
                
                # 系数可视化
                fig, ax = plt.subplots(figsize=(10, 6))
                coefs = significant_ols['系数'].sort_values()
                colors = ['green' if p < 0.01 else 'orange' 
                         for p in significant_ols.loc[coefs.index, 'P值']]
                ax.barh(range(len(coefs)), coefs.values, color=colors)
                ax.set_yticks(range(len(coefs)))
                ax.set_yticklabels(coefs.index)
                ax.set_xlabel('系数值')
                ax.set_title('OLS模型系数（显著特征）')
                ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
                ax.grid(True, alpha=0.3, axis='x')
                st.pyplot(fig)
                
                st.info("💡 OLS模型假设残差正态分布，但PM2.5数据通常不满足此假设，因此我们使用GLM模型进行优化")
            else:
                st.warning("⚠️ 没有发现显著特征（P < 0.05）")
        
        elif model_type_used == 'GLM' and 'glm_model' in st.session_state:
            glm = st.session_state['glm_model']
            
            # 模型摘要
            st.subheader("GLM模型统计摘要")
            st.text(glm.get_summary())
            
            # 显著特征
            st.subheader("显著特征（P < 0.05）")
            significant_features = glm.get_significant_features(alpha=0.05)
            
            if len(significant_features) > 0:
                st.dataframe(significant_features, use_container_width=True)
                
                # 系数可视化
                fig, ax = plt.subplots(figsize=(10, 6))
                coefs = significant_features['coef'].sort_values()
                colors = ['green' if p < 0.01 else 'orange' if p < 0.05 else 'red' 
                         for p in significant_features.loc[coefs.index, 'p_value']]
                ax.barh(range(len(coefs)), coefs.values, color=colors)
                ax.set_yticks(range(len(coefs)))
                ax.set_yticklabels(coefs.index)
                ax.set_xlabel('系数值')
                ax.set_title('GLM模型系数（显著特征）')
                ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
                ax.grid(True, alpha=0.3, axis='x')
                st.pyplot(fig)
                
                # 系数解释
                st.subheader("系数解释")
                for feature in significant_features.index:
                    if feature != 'const':
                        interpretation = glm.interpret_coefficient(feature)
                        with st.expander(f"📌 {feature}"):
                            st.text(interpretation)
            else:
                st.warning("⚠️ 没有发现显著特征（P < 0.05）")


def page_warning_center(df):
    """页面C：预警中心"""
    # 精美的页面标题 - 简洁专业
    st.markdown("""
    <div style="text-align: left; padding: 1.5rem 0; margin-bottom: 2rem; border-bottom: 2px solid #e9ecef;">
        <h1 style="color: #1a1a1a; font-size: 2rem; font-weight: 700; margin: 0 0 0.5rem 0;">
            ⚠️ 预警中心
        </h1>
        <p style="color: #6c757d; font-size: 0.95rem; margin: 0; font-weight: 400;">
            Warning Center & Predictive Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 检查PM2.5列是否存在
    if 'PM2.5' not in df.columns:
        st.error("❌ 数据中缺少PM2.5列！")
        st.info(f"**当前数据列名：** {', '.join(df.columns.tolist())}")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎲 HMM隐状态推断")
        st.info("💡 使用隐马尔可夫模型推断当前空气质量隐状态")
        
        # HMM模型参数
        n_states = st.slider("隐状态数量", min_value=2, max_value=5, value=3)
        
        if st.button("训练HMM模型", type="primary"):
            with st.spinner("正在训练HMM模型..."):
                # 准备观测值（使用多个特征）
                feature_cols = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
                available_cols = [col for col in feature_cols if col in df.columns]
                
                if len(available_cols) >= 2:
                    observations = df[available_cols].values
                    pm25_values = df['PM2.5'].values
                    
                    hmm_model = HMMModel(n_states=n_states)
                    hmm_model.fit(observations, pm25_values)
                    
                    st.session_state['hmm_model'] = hmm_model
                    st.session_state['hmm_observations'] = observations
                    st.success("✅ HMM模型训练完成")
                else:
                    st.error("数据中缺少足够的特征列")
        
        # 显示HMM结果
        if 'hmm_model' in st.session_state:
            hmm = st.session_state['hmm_model']
            
            # 当前状态
            st.markdown("#### 🎯 当前隐状态")
            if 'hmm_observations' in st.session_state:
                # 使用最后一条数据作为当前观测
                current_obs = st.session_state['hmm_observations'][-1:]
                state_info = hmm.predict_current_state(current_obs)
                
                # 状态显示（带颜色）
                state_colors = {
                    '优良': '🟢',
                    '轻度污染': '🟡',
                    '重度污染': '🔴'
                }
                state_emoji = state_colors.get(state_info['state_name'], '⚪')
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{state_emoji} {state_info['state_name']}</h3>
                    <p>当前空气质量隐状态</p>
                </div>
                """, unsafe_allow_html=True)
                
                # 状态概率
                st.write("**状态概率分布**")
                prob_df = pd.DataFrame([state_info['state_probabilities']])
                st.dataframe(prob_df, use_container_width=True)
            
            # 状态转移矩阵
            st.markdown("#### 📊 状态转移矩阵")
            trans_matrix = hmm.get_transition_matrix()
            st.dataframe(trans_matrix, use_container_width=True)
            
            # 可视化转移矩阵
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(trans_matrix, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
            ax.set_title('HMM状态转移矩阵')
            st.pyplot(fig)
    
    with col2:
        st.markdown("### 📈 ARIMA时间序列预测")
        st.info("💡 使用ARIMA模型预测未来24小时的PM2.5趋势。先进行平稳性检验（ADF），再进行时序分解")
        
        # 平稳性检验和时序分解
        if st.button("进行平稳性检验和时序分解", type="secondary"):
            with st.spinner("正在分析时间序列..."):
                if 'date' in df.index.names or isinstance(df.index, pd.DatetimeIndex):
                    pm25_series = df['PM2.5'].dropna()
                else:
                    pm25_series = df['PM2.5'].dropna().reset_index(drop=True)
                
                if len(pm25_series) > 100:
                    arima = ARIMAModel()
                    
                    # ADF平稳性检验
                    adf_result = arima.check_stationarity(pm25_series)
                    st.session_state['adf_result'] = adf_result
                    st.session_state['pm25_series_for_decompose'] = pm25_series
                    
                    # 时序分解
                    try:
                        decomposition = arima.decompose(pm25_series, period=7 if len(pm25_series) > 7 else None)
                        st.session_state['decomposition'] = decomposition
                    except:
                        pass
                    
                    st.success("✅ 平稳性检验和时序分解完成")
                else:
                    st.error("数据量不足")
        
        # 显示ADF检验结果
        if 'adf_result' in st.session_state:
            adf_result = st.session_state['adf_result']
            st.write("**ADF平稳性检验（Augmented Dickey-Fuller Test）**")
            st.metric("ADF统计量", f"{adf_result['adf_statistic']:.4f}")
            st.metric("P值", f"{adf_result['p_value']:.4f}")
            
            if adf_result['is_stationary']:
                st.success("✅ 序列是平稳的（p < 0.05）")
                st.info("💡 可以直接使用ARIMA模型，d=0")
            else:
                st.warning("⚠️ 序列非平稳（p ≥ 0.05）")
                st.info("💡 需要进行差分处理（d > 0）使序列平稳")
            
            st.write("**临界值**")
            for level, value in adf_result['critical_values'].items():
                st.write(f"- {level}: {value:.4f}")
        
        # 显示时序分解图
        if 'decomposition' in st.session_state:
            decomposition = st.session_state['decomposition']
            st.write("**时序分解（趋势、季节性、残差）**")
            
            fig, axes = plt.subplots(4, 1, figsize=(12, 10))
            decomposition.observed.plot(ax=axes[0], title='原始序列', color='blue')
            decomposition.trend.plot(ax=axes[1], title='趋势', color='green')
            decomposition.seasonal.plot(ax=axes[2], title='季节性', color='orange')
            decomposition.resid.plot(ax=axes[3], title='残差', color='red')
            
            for ax in axes:
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.info("💡 时序分解帮助我们理解数据的趋势和季节性模式，这是ARIMA建模的重要前提")
        
        forecast_steps = st.slider("预测步数", min_value=12, max_value=48, value=24, step=12)
        
        if st.button("运行ARIMA预测", type="primary"):
            with st.spinner("正在拟合ARIMA模型并生成预测..."):
                # 准备时间序列
                if 'date' in df.index.names or isinstance(df.index, pd.DatetimeIndex):
                    pm25_series = df['PM2.5'].dropna()
                else:
                    pm25_series = df['PM2.5'].dropna().reset_index(drop=True)
                
                if len(pm25_series) > 100:
                    arima = ARIMAModel()
                    arima.fit(pm25_series, auto_select=True)
                    
                    forecast = arima.predict(steps=forecast_steps, alpha=0.05)
                    
                    st.session_state['arima_model'] = arima
                    st.session_state['arima_forecast'] = forecast
                    st.session_state['arima_series'] = pm25_series
                    st.success("✅ ARIMA预测完成")
                else:
                    st.error("数据量不足，无法进行ARIMA建模")
        
        # 显示ARIMA预测结果
        if 'arima_model' in st.session_state and 'arima_forecast' in st.session_state:
            forecast = st.session_state['arima_forecast']
            series = st.session_state['arima_series']
            
            # 精美的预测图
            fig = go.Figure()
            
            # 历史数据
            fig.add_trace(go.Scatter(
                x=list(range(len(series))),
                y=series.values,
                mode='lines',
                name='历史数据',
                line=dict(
                    color='#667eea',
                    width=2,
                    shape='spline',
                    smoothing=1.3
                ),
                hovertemplate='<b>时间点</b>: %{x}<br><b>PM2.5</b>: %{y:.2f} μg/m³<extra></extra>'
            ))
            
            # 预测值
            forecast_start = len(series)
            forecast_x = list(range(forecast_start, forecast_start + len(forecast)))
            fig.add_trace(go.Scatter(
                x=forecast_x,
                y=forecast['forecast'],
                mode='lines+markers',
                name='预测值',
                line=dict(
                    color='#f093fb',
                    width=3,
                    dash='dash',
                    shape='spline',
                    smoothing=1.3
                ),
                marker=dict(size=6, color='#f093fb'),
                hovertemplate='<b>预测时间点</b>: %{x}<br><b>预测PM2.5</b>: %{y:.2f} μg/m³<extra></extra>'
            ))
            
            # 置信区间
            fig.add_trace(go.Scatter(
                x=forecast_x + forecast_x[::-1],
                y=list(forecast['upper']) + list(forecast['lower'])[::-1],
                fill='toself',
                fillcolor='rgba(240, 147, 251, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95%置信区间',
                showlegend=True,
                hoverinfo='skip'
            ))
            
            fig.update_layout(
                title=dict(
                    text=f"ARIMA预测（未来{forecast_steps}小时）",
                    font=dict(size=20, color='#2c3e50', family='Arial Black')
                ),
                xaxis=dict(
                    title="时间",
                    titlefont=dict(size=14, color='#2c3e50'),
                    gridcolor='rgba(128, 128, 128, 0.2)',
                    showgrid=True
                ),
                yaxis=dict(
                    title="PM2.5 (μg/m³)",
                    titlefont=dict(size=14, color='#2c3e50'),
                    gridcolor='rgba(128, 128, 128, 0.2)',
                    showgrid=True
                ),
                height=450,
                plot_bgcolor='rgba(255, 255, 255, 0.9)',
                paper_bgcolor='rgba(255, 255, 255, 0.9)',
                hovermode='x unified',
                font=dict(family="Arial", size=12),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 预测统计
            st.write("**预测统计**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("预测均值", f"{forecast['forecast'].mean():.2f}")
            with col2:
                st.metric("预测最大值", f"{forecast['forecast'].max():.2f}")
            with col3:
                st.metric("预测最小值", f"{forecast['forecast'].min():.2f}")


def main():
    """主函数"""
    # 精美的标题 - 简洁专业
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0 1rem 0; border-bottom: 2px solid #e9ecef; margin-bottom: 2rem;">
        <div class="main-header">
            <div style="font-size: 3rem; margin-bottom: 0.5rem; color: #0066cc;">🌫️</div>
            <div style="color: #1a1a1a; font-weight: 700; font-size: 2.5rem; margin-bottom: 0.5rem;">
                城市空气质量监测及归因预警系统
            </div>
            <div style="font-size: 1rem; color: #6c757d; margin-top: 0.5rem; font-weight: 400; letter-spacing: 0.5px;">
                Air Quality Monitoring & Attribution Early Warning System
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 侧边栏：数据上传和页面选择
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; margin-bottom: 1.5rem;">
            <h2 style="color: #f1f5f9; margin: 0; font-size: 1.3rem; font-weight: 600;">📁 数据管理</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # 数据上传
        uploaded_file = st.file_uploader("上传数据文件（CSV格式）", type=['csv'])
        
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            if df is not None:
                st.success(f"✅ 数据加载成功！共 {len(df)} 条记录，{len(df.columns)} 列")
                # 显示数据列名
                with st.expander("📋 查看数据列名"):
                    st.write("**所有列名：**")
                    st.write(", ".join(df.columns.tolist()))
                st.session_state['data'] = df
                # 清除之前的预处理数据，强制重新预处理
                if 'processed_data' in st.session_state:
                    del st.session_state['processed_data']
        elif 'data' not in st.session_state:
            # 如果没有上传文件，尝试加载默认数据
            default_path = "../Data/PRSA_data.csv"  # UCI数据集路径
            if os.path.exists(default_path):
                df = load_data(default_path)
                if df is not None:
                    st.session_state['data'] = df
                    st.info("📂 已加载默认数据集")
        
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; margin: 1.5rem 0;">
            <h2 style="color: #f1f5f9; margin: 0; font-size: 1.3rem; font-weight: 600;">📑 页面导航</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # 精美的页面选择
        page_options = {
            "数据洞察": "📊",
            "归因分析": "🔍",
            "预警中心": "⚠️"
        }
        
        page = st.radio(
            "选择页面",
            list(page_options.keys()),
            format_func=lambda x: f"{page_options[x]} {x}",
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # 添加项目信息
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.08); padding: 1rem; border-radius: 8px; margin-top: 2rem; border: 1px solid rgba(255, 255, 255, 0.1);">
            <p style="color: #cbd5e1; font-size: 0.85rem; margin: 0.5rem 0; line-height: 1.6;">
                <strong style="color: #f1f5f9;">项目代号:</strong><br>
                <span style="color: #94a3b8;">AirQuality-StatModel-2025</span>
            </p>
            <p style="color: #cbd5e1; font-size: 0.85rem; margin: 0.5rem 0; line-height: 1.6;">
                <strong style="color: #f1f5f9;">课程:</strong><br>
                <span style="color: #94a3b8;">统计分析与建模</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # 主内容区
    if 'data' in st.session_state:
        df = st.session_state['data'].copy()
        
        # 数据预处理
        if 'processed_data' not in st.session_state:
            with st.spinner("正在预处理数据..."):
                df_processed = preprocess_data(df)
                st.session_state['processed_data'] = df_processed
        else:
            df_processed = st.session_state['processed_data']
        
        # 根据选择的页面显示内容
        if page == "数据洞察":
            page_data_insight(df_processed)
        elif page == "归因分析":
            page_attribution_analysis(df_processed)
        elif page == "预警中心":
            page_warning_center(df_processed)
    else:
        st.warning("⚠️ 请先上传数据文件或确保默认数据文件存在")
        st.info("""
        **数据格式要求：**
        - CSV格式
        - 必须包含 'PM2.5' 列
        - 建议包含日期列（用于时间序列分析）
        - 建议包含气象因子：TEMP, PRES, DEWP, RAIN, WSPM, cbwd等
        """)


if __name__ == "__main__":
    main()

