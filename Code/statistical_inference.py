"""
统计推断模块
功能：假设检验、ANOVA、相关分析
符合课程要求：假设检验、方差分析、相关分析章节
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, f_oneway
import warnings
warnings.filterwarnings('ignore')


class StatisticalInference:
    """统计推断类"""
    
    def __init__(self, df):
        """
        初始化
        
        Parameters:
        -----------
        df : pd.DataFrame
            数据框
        """
        self.df = df.copy()
    
    def t_test(self, column='PM2.5', group_column='is_weekend'):
        """
        T检验：检验两组均值是否有显著差异
        应用场景：检验"周末"和"工作日"的空气质量是否有显著差异
        
        Parameters:
        -----------
        column : str
            要检验的数值列
        group_column : str
            分组列（二分类）
        
        Returns:
        --------
        dict : 检验结果
        """
        if group_column not in self.df.columns:
            # 如果没有分组列，尝试从日期创建
            if 'date' in self.df.index.names or isinstance(self.df.index, pd.DatetimeIndex):
                self.df['is_weekend'] = self.df.index.weekday >= 5
                group_column = 'is_weekend'
            else:
                raise ValueError(f"列 {group_column} 不存在，且无法从日期创建")
        
        groups = self.df[group_column].unique()
        if len(groups) != 2:
            raise ValueError("分组列必须只有两个类别")
        
        group1 = self.df[self.df[group_column] == groups[0]][column].dropna()
        group2 = self.df[self.df[group_column] == groups[1]][column].dropna()
        
        # 执行t检验
        t_stat, p_value = ttest_ind(group1, group2)
        
        result = {
            'test_name': 'Independent t-test',
            'groups': {groups[0]: len(group1), groups[1]: len(group2)},
            'group1_mean': group1.mean(),
            'group2_mean': group2.mean(),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'interpretation': f"{'有' if p_value < 0.05 else '无'}显著差异 (p={p_value:.4f})"
        }
        
        return result
    
    def anova_test(self, column='PM2.5', group_column='cbwd'):
        """
        方差分析（ANOVA）：检验多组均值是否有显著差异
        应用场景：检验不同风向下的污染均值是否存在显著差异
        
        Parameters:
        -----------
        column : str
            要检验的数值列
        group_column : str
            分组列（多分类）
        
        Returns:
        --------
        dict : 检验结果
        """
        if group_column not in self.df.columns:
            raise ValueError(f"列 {group_column} 不存在")
        
        groups = self.df[group_column].unique()
        if len(groups) < 2:
            raise ValueError("分组列至少需要两个类别")
        
        # 准备各组数据
        group_data = []
        group_stats = {}
        for group in groups:
            data = self.df[self.df[group_column] == group][column].dropna()
            group_data.append(data)
            group_stats[group] = {
                'count': len(data),
                'mean': data.mean(),
                'std': data.std()
            }
        
        # 执行ANOVA
        f_stat, p_value = f_oneway(*group_data)
        
        result = {
            'test_name': 'One-way ANOVA',
            'groups': group_stats,
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'interpretation': f"{'有' if p_value < 0.05 else '无'}显著差异 (p={p_value:.4f})"
        }
        
        return result
    
    def correlation_analysis(self, columns=None, method='pearson'):
        """
        相关分析：计算变量间的相关系数
        应用场景：筛选自变量，剔除多重共线性
        
        Parameters:
        -----------
        columns : list, optional
            要分析的列名列表，默认使用所有数值列
        method : str
            方法：'pearson'（线性相关）或 'spearman'（秩相关）
        
        Returns:
        --------
        pd.DataFrame : 相关系数矩阵
        """
        if columns is None:
            # 选择数值列
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            columns = numeric_cols
        
        # 计算相关系数矩阵
        corr_matrix = self.df[columns].corr(method=method)
        
        return corr_matrix
    
    def detect_multicollinearity(self, threshold=0.8):
        """
        检测多重共线性
        应用场景：如果温度和气压高度相关（>0.8），只能留一个
        
        Parameters:
        -----------
        threshold : float
            相关系数阈值，默认0.8
        
        Returns:
        --------
        list : 高相关变量对列表
        """
        corr_matrix = self.correlation_analysis()
        
        # 找出高相关对
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    high_corr_pairs.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        return high_corr_pairs


if __name__ == "__main__":
    # 测试代码
    print("统计推断模块已就绪")

