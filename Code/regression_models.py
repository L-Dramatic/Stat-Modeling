"""
其他回归模型模块
功能：实现Ridge、Lasso回归，提供模型对比功能
符合课程要求：其他回归模型章节
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 导入现有GLM模块
try:
    from glm_model import GLMModel
except ImportError:
    GLMModel = None


class RegressionModels:
    """统一回归模型类"""
    
    def __init__(self):
        """初始化回归模型"""
        self.ols_model = None
        self.ridge_model = None
        self.lasso_model = None
        self.glm_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def fit_ols(self, X, y, add_constant=True):
        """
        拟合OLS基准模型
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            特征变量
        y : pd.Series or np.array
            目标变量
        add_constant : bool
            是否添加常数项
            
        Returns:
        --------
        RegressionResults : statsmodels回归结果
        """
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
            self.feature_names = X_df.columns.tolist()
        else:
            X_df = pd.DataFrame(X)
            self.feature_names = [f'特征{i+1}' for i in range(X_df.shape[1])]
        
        if isinstance(y, pd.Series):
            y_series = y.copy()
        else:
            y_series = pd.Series(y)
        
        # 添加常数项
        if add_constant:
            X_df = sm.add_constant(X_df)
        
        # 拟合OLS模型
        self.ols_model = sm.OLS(y_series, X_df).fit()
        
        return self.ols_model
    
    def fit_ridge(self, X, y, alpha=None, cv=True):
        """
        拟合Ridge回归模型
        
        Ridge回归通过L2正则化处理多重共线性问题
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            特征变量
        y : pd.Series or np.array
            目标变量
        alpha : float, optional
            正则化参数（如果None，使用交叉验证选择）
        cv : bool
            是否使用交叉验证选择alpha
            
        Returns:
        --------
        Ridge : 拟合后的模型
        """
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            self.feature_names = X.columns.tolist()
        else:
            X_array = np.array(X)
            self.feature_names = [f'特征{i+1}' for i in range(X_array.shape[1])]
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = np.array(y)
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X_array)
        
        if cv:
            # 使用交叉验证选择最优alpha
            alphas = np.logspace(-4, 1, 50)
            self.ridge_model = RidgeCV(alphas=alphas, cv=5)
        else:
            if alpha is None:
                alpha = 1.0
            self.ridge_model = Ridge(alpha=alpha)
        
        self.ridge_model.fit(X_scaled, y_array)
        
        return self.ridge_model
    
    def fit_lasso(self, X, y, alpha=None, cv=True):
        """
        拟合Lasso回归模型（用于特征选择）
        
        Lasso回归通过L1正则化进行特征选择，将不重要变量的系数压缩为0
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            特征变量
        y : pd.Series or np.array
            目标变量
        alpha : float, optional
            正则化参数（如果None，使用交叉验证选择）
        cv : bool
            是否使用交叉验证选择alpha
            
        Returns:
        --------
        Lasso : 拟合后的模型
        """
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            self.feature_names = X.columns.tolist()
        else:
            X_array = np.array(X)
            self.feature_names = [f'特征{i+1}' for i in range(X_array.shape[1])]
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = np.array(y)
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X_array)
        
        if cv:
            # 使用交叉验证选择最优alpha
            alphas = np.logspace(-4, 1, 50)
            self.lasso_model = LassoCV(alphas=alphas, cv=5, max_iter=2000)
        else:
            if alpha is None:
                alpha = 0.1
            self.lasso_model = Lasso(alpha=alpha, max_iter=2000)
        
        self.lasso_model.fit(X_scaled, y_array)
        
        return self.lasso_model
    
    def fit_glm(self, X, y, family='gamma'):
        """
        拟合GLM模型（调用现有GLM模块）
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            特征变量
        y : pd.Series or np.array
            目标变量
        family : str
            分布族（默认'gamma'）
            
        Returns:
        --------
        GLMModel : 拟合后的模型
        """
        if GLMModel is None:
            raise ImportError("GLMModel模块未找到，请确保glm_model.py存在")
        
        self.glm_model = GLMModel(family=family)
        self.glm_model.fit(X, y)
        
        return self.glm_model
    
    def get_lasso_selected_features(self, threshold=1e-5):
        """
        获取Lasso筛选出的重要特征
        
        Parameters:
        -----------
        threshold : float
            系数阈值，绝对值小于此值的特征被认为不重要
            
        Returns:
        --------
        pd.DataFrame : 重要特征及其系数
        """
        if self.lasso_model is None:
            raise ValueError("Lasso模型尚未拟合，请先调用fit_lasso()")
        
        coef = self.lasso_model.coef_
        
        # 筛选重要特征
        important_mask = np.abs(coef) > threshold
        important_features = np.array(self.feature_names)[important_mask]
        important_coef = coef[important_mask]
        
        result = pd.DataFrame({
            '特征': important_features,
            '系数': important_coef
        }).sort_values('系数', key=abs, ascending=False)
        
        return result
    
    def compare_models(self, X, y):
        """
        对比所有回归模型的性能
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            特征变量
        y : pd.Series or np.array
            目标变量
            
        Returns:
        --------
        pd.DataFrame : 模型对比表格（包含AIC、BIC、R²等）
        """
        results = []
        
        # OLS模型
        if self.ols_model is not None:
            y_pred = self.ols_model.predict(sm.add_constant(X) if isinstance(X, pd.DataFrame) else sm.add_constant(X))
            r2 = self.ols_model.rsquared
            aic = self.ols_model.aic
            bic = self.ols_model.bic
            results.append({
                '模型': 'OLS',
                'R²': r2,
                'AIC': aic,
                'BIC': bic
            })
        
        # Ridge模型
        if self.ridge_model is not None:
            if isinstance(X, pd.DataFrame):
                X_scaled = self.scaler.transform(X.values)
            else:
                X_scaled = self.scaler.transform(X)
            y_pred = self.ridge_model.predict(X_scaled)
            from sklearn.metrics import r2_score
            r2 = r2_score(y, y_pred)
            # 简化计算AIC/BIC
            n = len(y)
            mse = np.mean((y - y_pred) ** 2)
            k = len(self.ridge_model.coef_) + 1
            aic = n * np.log(mse) + 2 * k
            bic = n * np.log(mse) + k * np.log(n)
            results.append({
                '模型': 'Ridge',
                'R²': r2,
                'AIC': aic,
                'BIC': bic
            })
        
        # Lasso模型
        if self.lasso_model is not None:
            if isinstance(X, pd.DataFrame):
                X_scaled = self.scaler.transform(X.values)
            else:
                X_scaled = self.scaler.transform(X)
            y_pred = self.lasso_model.predict(X_scaled)
            from sklearn.metrics import r2_score
            r2 = r2_score(y, y_pred)
            n = len(y)
            mse = np.mean((y - y_pred) ** 2)
            k = np.sum(np.abs(self.lasso_model.coef_) > 1e-5) + 1  # 非零系数数量
            aic = n * np.log(mse) + 2 * k
            bic = n * np.log(mse) + k * np.log(n)
            results.append({
                '模型': 'Lasso',
                'R²': r2,
                'AIC': aic,
                'BIC': bic
            })
        
        # GLM模型
        if self.glm_model is not None and self.glm_model.results is not None:
            y_pred = self.glm_model.predict(X)
            from sklearn.metrics import r2_score
            r2 = r2_score(y, y_pred)
            aic = self.glm_model.results.aic
            bic = self.glm_model.results.bic
            results.append({
                '模型': 'GLM (Gamma)',
                'R²': r2,
                'AIC': aic,
                'BIC': bic
            })
        
        if not results:
            return pd.DataFrame()
        
        return pd.DataFrame(results)


if __name__ == "__main__":
    # 测试代码
    print("回归模型模块已就绪")
    print("包含模型：OLS, Ridge, Lasso, GLM")


