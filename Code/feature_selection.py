"""
特征选择模块
功能：使用Lasso回归进行特征筛选，识别关键变量
符合课程要求：数据预处理与特征选择章节
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, Lasso

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class FeatureSelector:
    """特征选择类"""
    
    def __init__(self):
        """初始化特征选择器"""
        self.lasso_model = None
        self.scaler = StandardScaler()
        self.selected_features = None
        self.feature_importance = None
    
    def lasso_selection(self, X, y, alpha_range=None, cv=5):
        """
        使用Lasso回归进行特征筛选
        
        Lasso通过L1正则化将不重要变量的系数压缩为0，实现特征选择
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            特征变量
        y : pd.Series or np.array
            目标变量
        alpha_range : array-like, optional
            正则化参数范围（如果None，自动生成）
        cv : int
            交叉验证折数
            
        Returns:
        --------
        dict : 包含筛选结果的字典
        """
        # 转换为数组
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            feature_names = X.columns.tolist()
        else:
            X_array = np.array(X)
            feature_names = [f'特征{i+1}' for i in range(X_array.shape[1])]
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = np.array(y)
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X_array)
        
        # 设置alpha范围
        if alpha_range is None:
            alpha_range = np.logspace(-4, 1, 50)
        
        # 使用交叉验证选择最优alpha
        self.lasso_model = LassoCV(alphas=alpha_range, cv=cv, max_iter=2000, random_state=42)
        self.lasso_model.fit(X_scaled, y_array)
        
        # 获取系数
        coef = self.lasso_model.coef_
        optimal_alpha = self.lasso_model.alpha_
        
        # 筛选重要特征（系数不为0）
        selected_mask = np.abs(coef) > 1e-5
        selected_features = np.array(feature_names)[selected_mask]
        selected_coef = coef[selected_mask]
        
        # 创建特征重要性DataFrame
        self.feature_importance = pd.DataFrame({
            '特征': feature_names,
            '系数': coef,
            '重要性（绝对值）': np.abs(coef),
            '是否选中': selected_mask
        }).sort_values('重要性（绝对值）', ascending=False)
        
        self.selected_features = selected_features.tolist()
        
        result = {
            'selected_features': self.selected_features,
            'optimal_alpha': optimal_alpha,
            'feature_importance': self.feature_importance,
            'n_selected': len(selected_features),
            'n_total': len(feature_names)
        }
        
        return result
    
    def get_feature_importance(self):
        """
        获取特征重要性排序
        
        Returns:
        --------
        pd.DataFrame : 特征重要性表格
        """
        if self.feature_importance is None:
            raise ValueError("尚未进行特征选择，请先调用lasso_selection()")
        
        return self.feature_importance
    
    def plot_feature_importance(self, top_n=10, figsize=(10, 6)):
        """
        绘制特征重要性条形图
        
        Parameters:
        -----------
        top_n : int
            显示前N个重要特征
        figsize : tuple
            图形大小
            
        Returns:
        --------
        fig, ax : matplotlib图形对象
        """
        if self.feature_importance is None:
            raise ValueError("尚未进行特征选择，请先调用lasso_selection()")
        
        top_features = self.feature_importance.head(top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        colors = ['green' if selected else 'gray' for selected in top_features['是否选中']]
        
        ax.barh(range(len(top_features)), top_features['重要性（绝对值）'], color=colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['特征'])
        ax.set_xlabel('特征重要性（系数绝对值）')
        ax.set_title(f'特征重要性排序（Top {top_n}）')
        ax.grid(True, alpha=0.3, axis='x')
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='选中特征'),
            Patch(facecolor='gray', label='未选中特征')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        
        return fig, ax
    
    def plot_regularization_path(self, X, y, alpha_range=None, figsize=(10, 6)):
        """
        绘制正则化路径图（展示不同alpha下系数的变化）
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            特征变量
        y : pd.Series or np.array
            目标变量
        alpha_range : array-like, optional
            正则化参数范围
        figsize : tuple
            图形大小
            
        Returns:
        --------
        fig, ax : matplotlib图形对象
        """
        # 转换为数组
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            feature_names = X.columns.tolist()
        else:
            X_array = np.array(X)
            feature_names = [f'特征{i+1}' for i in range(X_array.shape[1])]
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = np.array(y)
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X_array)
        
        # 设置alpha范围
        if alpha_range is None:
            alpha_range = np.logspace(-4, 1, 100)
        
        # 计算不同alpha下的系数
        coef_path = []
        for alpha in alpha_range:
            lasso = Lasso(alpha=alpha, max_iter=2000)
            lasso.fit(X_scaled, y_array)
            coef_path.append(lasso.coef_)
        
        coef_path = np.array(coef_path)
        
        # 绘图
        fig, ax = plt.subplots(figsize=figsize)
        
        for i in range(coef_path.shape[1]):
            ax.plot(alpha_range, coef_path[:, i], label=feature_names[i] if i < len(feature_names) else f'特征{i+1}')
        
        ax.set_xscale('log')
        ax.set_xlabel('正则化参数 (alpha)')
        ax.set_ylabel('系数值')
        ax.set_title('Lasso正则化路径')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig, ax
    
    def get_selected_features_list(self):
        """
        获取筛选出的特征列表
        
        Returns:
        --------
        list : 选中特征的名称列表
        """
        if self.selected_features is None:
            raise ValueError("尚未进行特征选择，请先调用lasso_selection()")
        
        return self.selected_features.copy()


if __name__ == "__main__":
    # 测试代码
    print("特征选择模块已就绪")
    print("功能：Lasso特征筛选、特征重要性分析、正则化路径可视化")

