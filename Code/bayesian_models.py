"""
贝叶斯模型模块
功能：实现贝叶斯回归和贝叶斯分类，提供后验分布分析
符合课程要求：贝叶斯章节
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')


class BayesianModels:
    """贝叶斯模型类"""
    
    def __init__(self):
        """初始化贝叶斯模型"""
        self.bayesian_ridge_model = None
        self.bayesian_classifier = None
        self.is_fitted = False
    
    def fit_bayesian_regression(self, X, y):
        """
        拟合贝叶斯岭回归模型
        
        贝叶斯回归的优势：
        1. 提供参数的后验分布
        2. 自动处理过拟合（通过正则化）
        3. 量化不确定性（可信区间）
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            特征变量
        y : pd.Series or np.array
            目标变量（PM2.5）
            
        Returns:
        --------
        BayesianRidge : 拟合后的模型
        """
        # 转换为数组
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.array(X)
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = np.array(y)
        
        # 确保y为正值
        if (y_array <= 0).any():
            y_array = y_array + 1e-6
        
        # 拟合贝叶斯岭回归
        self.bayesian_ridge_model = BayesianRidge(
            max_iter=300,  # 新版本scikit-learn使用max_iter而不是n_iter
            compute_score=True,
            alpha_1=1e-6,
            alpha_2=1e-6,
            lambda_1=1e-6,
            lambda_2=1e-6
        )
        self.bayesian_ridge_model.fit(X_array, y_array)
        self.is_fitted = True
        
        return self.bayesian_ridge_model
    
    def predict_bayesian_regression(self, X, return_std=True):
        """
        使用贝叶斯回归模型预测
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            特征变量
        return_std : bool
            是否返回预测的标准差（不确定性）
            
        Returns:
        --------
        tuple or np.array : 预测值和标准差（如果return_std=True）
        """
        if self.bayesian_ridge_model is None:
            raise ValueError("贝叶斯回归模型尚未拟合，请先调用fit_bayesian_regression()")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.array(X)
        
        if return_std:
            y_pred, y_std = self.bayesian_ridge_model.predict(X_array, return_std=True)
            return y_pred, y_std
        else:
            y_pred = self.bayesian_ridge_model.predict(X_array)
            return y_pred
    
    def get_posterior_distribution(self):
        """
        获取参数的后验分布信息
        
        Returns:
        --------
        dict : 包含后验分布信息的字典
        """
        if self.bayesian_ridge_model is None:
            raise ValueError("贝叶斯回归模型尚未拟合")
        
        # 贝叶斯岭回归的参数
        # alpha_ 和 lambda_ 是超参数的后验分布参数
        # coef_ 是系数的后验均值
        
        # 处理sigma_：如果是矩阵，取对角线；如果是向量，直接使用
        sigma = self.bayesian_ridge_model.sigma_
        if sigma.ndim > 1:
            # 协方差矩阵，取对角线元素
            coef_std = np.sqrt(np.diag(sigma))
        else:
            # 已经是方差向量
            coef_std = np.sqrt(sigma)
        
        result = {
            'coefficients_mean': self.bayesian_ridge_model.coef_,
            'coefficients_std': coef_std,
            'alpha': self.bayesian_ridge_model.alpha_,
            'lambda': self.bayesian_ridge_model.lambda_,
            'score': self.bayesian_ridge_model.scores_[-1] if hasattr(self.bayesian_ridge_model, 'scores_') else None
        }
        
        return result
    
    def get_credible_intervals(self, X, alpha=0.05):
        """
        获取预测的可信区间（Credible Intervals）
        
        注意：可信区间（贝叶斯）vs 置信区间（频率学派）
        - 可信区间：参数落在区间内的概率
        - 置信区间：重复实验时区间包含参数的概率
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            特征变量
        alpha : float
            显著性水平（默认0.05，即95%可信区间）
            
        Returns:
        --------
        pd.DataFrame : 包含预测值、下界、上界的DataFrame
        """
        y_pred, y_std = self.predict_bayesian_regression(X, return_std=True)
        
        # 使用正态分布假设计算可信区间
        from scipy import stats
        z_score = stats.norm.ppf(1 - alpha/2)
        
        lower = y_pred - z_score * y_std
        upper = y_pred + z_score * y_std
        
        result = pd.DataFrame({
            'prediction': y_pred,
            'lower': lower,
            'upper': upper,
            'std': y_std
        })
        
        return result
    
    def plot_posterior(self, feature_names=None, figsize=(12, 6)):
        """
        可视化参数后验分布
        
        Parameters:
        -----------
        feature_names : list, optional
            特征名称
        figsize : tuple
            图形大小
            
        Returns:
        --------
        fig, ax : matplotlib图形对象
        """
        if self.bayesian_ridge_model is None:
            raise ValueError("贝叶斯回归模型尚未拟合")
        
        posterior = self.get_posterior_distribution()
        coef_mean = np.array(posterior['coefficients_mean']).flatten()
        coef_std = np.array(posterior['coefficients_std']).flatten()
        
        if feature_names is None:
            feature_names = [f'特征{i+1}' for i in range(len(coef_mean))]
        else:
            # 确保feature_names是列表
            feature_names = list(feature_names)
        
        # 确保长度匹配
        min_len = min(len(coef_mean), len(coef_std), len(feature_names))
        coef_mean = coef_mean[:min_len]
        coef_std = coef_std[:min_len]
        feature_names = feature_names[:min_len]
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 1. 系数后验均值
        axes[0].barh(feature_names, coef_mean)
        axes[0].axvline(0, color='r', linestyle='--', linewidth=1)
        axes[0].set_xlabel('系数值')
        axes[0].set_title('系数后验均值')
        axes[0].grid(True, alpha=0.3)
        
        # 2. 系数不确定性（标准差）
        axes[1].barh(feature_names, coef_std, color='orange')
        axes[1].set_xlabel('标准差')
        axes[1].set_title('系数不确定性（后验标准差）')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig, axes
    
    def compare_with_frequentist(self, frequentist_coef, frequentist_std=None):
        """
        与频率学派方法对比
        
        Parameters:
        -----------
        frequentist_coef : array-like
            频率学派方法的系数
        frequentist_std : array-like, optional
            频率学派方法的标准误
            
        Returns:
        --------
        pd.DataFrame : 对比表格
        """
        if self.bayesian_ridge_model is None:
            raise ValueError("贝叶斯回归模型尚未拟合")
        
        posterior = self.get_posterior_distribution()
        
        comparison = pd.DataFrame({
            '贝叶斯均值': posterior['coefficients_mean'],
            '贝叶斯标准差': posterior['coefficients_std'],
            '频率学派系数': np.array(frequentist_coef)
        })
        
        if frequentist_std is not None:
            comparison['频率学派标准误'] = np.array(frequentist_std)
        
        comparison['差异'] = comparison['贝叶斯均值'] - comparison['频率学派系数']
        
        return comparison
    
    def fit_bayesian_classification(self, X, y):
        """
        拟合贝叶斯分类器（朴素贝叶斯）
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            特征变量
        y : pd.Series or np.array
            分类标签
            
        Returns:
        --------
        GaussianNB : 拟合后的模型
        """
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.array(X)
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = np.array(y)
        
        self.bayesian_classifier = GaussianNB()
        self.bayesian_classifier.fit(X_array, y_array)
        self.is_fitted = True
        
        return self.bayesian_classifier
    
    def get_summary(self):
        """
        获取贝叶斯模型摘要
        
        Returns:
        --------
        dict : 模型摘要信息
        """
        if self.bayesian_ridge_model is None:
            raise ValueError("贝叶斯回归模型尚未拟合")
        
        posterior = self.get_posterior_distribution()
        
        summary = {
            '模型类型': 'Bayesian Ridge Regression',
            '系数数量': len(posterior['coefficients_mean']),
            'alpha (正则化参数)': posterior['alpha'],
            'lambda (正则化参数)': posterior['lambda'],
            '模型得分': posterior['score']
        }
        
        return summary


if __name__ == "__main__":
    # 测试代码
    print("贝叶斯模型模块已就绪")
    print("包含模型：Bayesian Ridge Regression, Bayesian Classification")

