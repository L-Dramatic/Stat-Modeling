"""
ARIMA时间序列模型模块
功能：捕捉季节性，预测未来短期趋势
符合课程要求：时间序列分析章节
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')


class ARIMAModel:
    """ARIMA时间序列模型类"""
    
    def __init__(self):
        """初始化ARIMA模型"""
        self.model = None
        self.results = None
        self.order = None
        self.seasonal_order = None
    
    def check_stationarity(self, series, alpha=0.05):
        """
        检验序列平稳性（ADF检验）
        
        Parameters:
        -----------
        series : pd.Series
            时间序列
        alpha : float
            显著性水平
        
        Returns:
        --------
        dict : 检验结果
        """
        result = adfuller(series.dropna())
        
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'is_stationary': result[1] < alpha,
            'critical_values': result[4]
        }
    
    def decompose(self, series, model='additive', period=None):
        """
        时序分解（趋势、季节性、残差）
        
        Parameters:
        -----------
        series : pd.Series
            时间序列
        model : str
            模型类型：'additive'（加法）或 'multiplicative'（乘法）
        period : int, optional
            季节性周期，默认自动检测
        
        Returns:
        --------
        DecomposeResult : 分解结果
        """
        if period is None:
            # 尝试检测周期（假设日数据，周期为7天或365天）
            if len(series) > 365:
                period = 365
            elif len(series) > 7:
                period = 7
            else:
                period = None
        
        if period:
            decomposition = seasonal_decompose(
                series.dropna(),
                model=model,
                period=period,
                extrapolate_trend='freq'
            )
        else:
            decomposition = seasonal_decompose(
                series.dropna(),
                model=model,
                extrapolate_trend='freq'
            )
        
        return decomposition
    
    def auto_arima(self, series, max_p=3, max_d=2, max_q=3, seasonal=True, period=7):
        """
        自动选择ARIMA参数（简化版，实际可用pmdarima库）
        
        Parameters:
        -----------
        series : pd.Series
            时间序列
        max_p, max_d, max_q : int
            最大p, d, q值
        seasonal : bool
            是否使用季节性ARIMA
        period : int
            季节性周期
        
        Returns:
        --------
        tuple : (order, seasonal_order)
        """
        # 如果数据量太大，先降采样以加快速度
        if len(series) > 5000:
            # 降采样到5000个点
            step = len(series) // 5000
            series = series.iloc[::step]
        
        best_aic = np.inf
        best_order = None
        best_seasonal_order = None
        
        # 优化搜索范围：减少组合数量
        # 限制搜索范围以提高速度
        max_p = min(max_p, 2)  # 限制最大p为2
        max_d = min(max_d, 2)  # 限制最大d为2
        max_q = min(max_q, 2)  # 限制最大q为2
        
        total_combinations = (max_p + 1) * (max_d + 1) * (max_q + 1)
        
        # 简化搜索（实际应该用更智能的方法）
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        if seasonal:
                            model = ARIMA(series, order=(p, d, q), 
                                         seasonal_order=(1, 1, 1, period))
                        else:
                            model = ARIMA(series, order=(p, d, q))
                        
                        # 使用更快的拟合方法
                        fitted_model = model.fit(method_kwargs={"maxiter": 50})  # 限制迭代次数
                        aic = fitted_model.aic
                        
                        if aic < best_aic:
                            best_aic = aic
                            best_order = (p, d, q)
                            if seasonal:
                                best_seasonal_order = (1, 1, 1, period)
                    except:
                        continue
        
        return best_order, best_seasonal_order
    
    def fit(self, series, order=None, seasonal_order=None, auto_select=True):
        """
        拟合ARIMA模型
        
        Parameters:
        -----------
        series : pd.Series
            时间序列
        order : tuple, optional
            (p, d, q) 参数
        seasonal_order : tuple, optional
            (P, D, Q, s) 季节性参数
        auto_select : bool
            是否自动选择参数
        """
        if auto_select and order is None:
            order, seasonal_order = self.auto_arima(series)
        
        if order is None:
            # 默认参数
            order = (1, 1, 1)
        
        self.order = order
        
        if seasonal_order:
            self.seasonal_order = seasonal_order
            self.model = ARIMA(series, order=order, seasonal_order=seasonal_order)
        else:
            self.model = ARIMA(series, order=order)
        
        self.results = self.model.fit()
        
        return self.results
    
    def predict(self, steps=24, alpha=0.05):
        """
        预测未来值
        
        Parameters:
        -----------
        steps : int
            预测步数（例如：24小时）
        alpha : float
            置信水平
        
        Returns:
        --------
        pd.DataFrame : 包含预测值、置信区间的DataFrame
        """
        if self.results is None:
            raise ValueError("模型尚未拟合，请先调用fit()方法")
        
        forecast = self.results.get_forecast(steps=steps)
        conf_int = forecast.conf_int(alpha=alpha)
        
        # 获取置信区间（兼容不同版本的statsmodels列名格式）
        # conf_int通常是两列的DataFrame，第一列是下界，第二列是上界
        # 列名可能是：['lower 0.95', 'upper 0.95'] 或 [0, 1] 或其他格式
        try:
            # 尝试直接使用列名（如果格式是 'lower 0.95' 和 'upper 0.95'）
            conf_level = 1 - alpha
            lower_col = f'lower {conf_level:.2f}'
            upper_col = f'upper {conf_level:.2f}'
            
            if lower_col in conf_int.columns and upper_col in conf_int.columns:
                lower = conf_int[lower_col]
                upper = conf_int[upper_col]
            else:
                # 如果列名格式不同，使用第一列和第二列
                lower = conf_int.iloc[:, 0]
                upper = conf_int.iloc[:, 1]
        except Exception:
            # 如果出错，使用第一列和第二列
            lower = conf_int.iloc[:, 0]
            upper = conf_int.iloc[:, 1]
        
        forecast_df = pd.DataFrame({
            'forecast': forecast.predicted_mean,
            'lower': lower,
            'upper': upper
        })
        
        return forecast_df
    
    def get_residuals(self):
        """
        获取残差（用于残差检验）
        
        Returns:
        --------
        pd.Series : 残差序列
        """
        if self.results is None:
            raise ValueError("模型尚未拟合，请先调用fit()方法")
        
        return self.results.resid
    
    def get_summary(self):
        """
        获取模型摘要
        
        Returns:
        --------
        str : 模型摘要文本
        """
        if self.results is None:
            raise ValueError("模型尚未拟合，请先调用fit()方法")
        
        return str(self.results.summary())


if __name__ == "__main__":
    # 测试代码
    print("ARIMA模型模块已就绪")

