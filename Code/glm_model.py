"""
广义线性模型（GLM）模块
功能：使用Gamma分布族建立回归模型
符合课程要求：广义线性回归章节
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.families import Gamma
from statsmodels.genmod.families.links import log
import warnings
warnings.filterwarnings('ignore')


class GLMModel:
    """广义线性模型类（Gamma分布族）"""
    
    def __init__(self, family='gamma', link='log'):
        """
        初始化GLM模型
        
        Parameters:
        -----------
        family : str
            分布族：'gamma'（Gamma分布）
        link : str
            链接函数：'log'（对数链接）
        """
        self.family = family
        self.link = link
        self.model = None
        self.results = None
        self.feature_names = None
    
    def fit(self, X, y, add_constant=True):
        """
        拟合GLM模型
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            自变量
        y : pd.Series or np.array
            因变量（PM2.5）
        add_constant : bool
            是否添加常数项
        """
        # 转换为DataFrame
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X)
        
        if isinstance(y, pd.Series):
            y_series = y.copy()
        else:
            y_series = pd.Series(y)
        
        # 确保y为正值（Gamma分布要求）
        if (y_series <= 0).any():
            y_series = y_series + 1e-6
        
        # 添加常数项
        if add_constant:
            X_df = sm.add_constant(X_df)
        
        self.feature_names = X_df.columns.tolist()
        
        # 定义分布族和链接函数
        if self.family == 'gamma':
            family = Gamma(link=log())
        else:
            raise ValueError(f"不支持的分布族: {self.family}")
        
        # 拟合模型
        self.model = sm.GLM(y_series, X_df, family=family)
        self.results = self.model.fit()
        
        return self.results
    
    def predict(self, X, add_constant=True):
        """
        预测
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            自变量
        
        Returns:
        --------
        np.array : 预测值
        """
        if self.results is None:
            raise ValueError("模型尚未拟合，请先调用fit()方法")
        
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X)
        
        if add_constant:
            X_df = sm.add_constant(X_df)
        
        predictions = self.results.predict(X_df)
        return predictions
    
    def get_summary(self):
        """
        获取模型摘要（统计报表）
        重点展示：P值、置信区间、系数解释
        
        Returns:
        --------
        str : 模型摘要文本
        """
        if self.results is None:
            raise ValueError("模型尚未拟合，请先调用fit()方法")
        
        return str(self.results.summary())
    
    def get_significant_features(self, alpha=0.05):
        """
        获取显著特征（P < alpha）
        
        Parameters:
        -----------
        alpha : float
            显著性水平，默认0.05
        
        Returns:
        --------
        pd.DataFrame : 显著特征及其系数
        """
        if self.results is None:
            raise ValueError("模型尚未拟合，请先调用fit()方法")
        
        summary_df = pd.DataFrame({
            'coef': self.results.params,
            'std_err': self.results.bse,
            'z_value': self.results.tvalues,
            'p_value': self.results.pvalues,
            'conf_lower': self.results.conf_int()[0],
            'conf_upper': self.results.conf_int()[1]
        })
        
        # 筛选显著特征
        significant = summary_df[summary_df['p_value'] < alpha].copy()
        significant = significant.sort_values('p_value')
        
        return significant
    
    def interpret_coefficient(self, feature_name):
        """
        解释系数含义
        例如："系数0.5代表风速每增加1单位，PM2.5变化多少"
        
        Parameters:
        -----------
        feature_name : str
            特征名称
        
        Returns:
        --------
        str : 系数解释文本
        """
        if self.results is None:
            raise ValueError("模型尚未拟合，请先调用fit()方法")
        
        if feature_name not in self.results.params.index:
            raise ValueError(f"特征 {feature_name} 不存在")
        
        coef = self.results.params[feature_name]
        p_value = self.results.pvalues[feature_name]
        
        # 对于对数链接函数，系数解释为：exp(coef) - 1 是相对变化
        relative_change = (np.exp(coef) - 1) * 100
        
        interpretation = f"""
        特征: {feature_name}
        系数: {coef:.4f}
        P值: {p_value:.4f}
        显著性: {'显著' if p_value < 0.05 else '不显著'}
        
        解释: 在控制其他变量不变的情况下，{feature_name}每增加1个单位，
        PM2.5的期望值将变化约 {relative_change:.2f}%
        （使用对数链接函数，相对变化 = exp(系数) - 1）
        """
        
        return interpretation


if __name__ == "__main__":
    # 测试代码
    print("GLM模型模块已就绪")

