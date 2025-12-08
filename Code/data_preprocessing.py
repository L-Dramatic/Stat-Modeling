"""
数据预处理模块
功能：数据清洗、插值填补、异常值剔除、分布拟合
符合课程要求：数据预处理章节
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import gamma, lognorm, norm
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, data_path=None, df=None):
        """
        初始化预处理器
        
        Parameters:
        -----------
        data_path : str, optional
            数据文件路径
        df : pd.DataFrame, optional
            直接传入的DataFrame
        """
        if df is not None:
            self.df = df.copy()
        elif data_path:
            self.df = pd.read_csv(data_path)
        else:
            raise ValueError("必须提供data_path或df参数")
        
        # 尝试识别日期列（不区分大小写）
        date_col = None
        for col in self.df.columns:
            if col.lower() in ['date', 'datetime', 'time', 'timestamp']:
                date_col = col
                break
        
        if date_col:
            try:
                self.df[date_col] = pd.to_datetime(self.df[date_col])
                if date_col != 'date':
                    self.df.rename(columns={date_col: 'date'}, inplace=True)
                # 设置日期为索引
                self.df.set_index('date', inplace=True)
            except:
                # 如果日期转换失败，不设置索引
                pass
    
    def handle_missing_values(self, method='interpolation', **kwargs):
        """
        处理缺失值
        
        Parameters:
        -----------
        method : str
            方法：'interpolation'（插值）或 'drop'（删除）
        """
        if method == 'interpolation':
            # 线性插值
            self.df = self.df.interpolate(method='linear', limit_direction='both')
            # 如果还有缺失值，用前向填充（兼容新版本pandas）
            self.df = self.df.ffill().bfill()
        elif method == 'drop':
            self.df = self.df.dropna()
        
        return self.df
    
    def remove_outliers(self, column='PM2.5', method='3sigma', **kwargs):
        """
        异常值剔除（3σ原则）
        
        Parameters:
        -----------
        column : str
            要处理的列名
        method : str
            方法：'3sigma'（3倍标准差）或 'iqr'（四分位距）
        """
        if column not in self.df.columns:
            return self.df
        
        if method == '3sigma':
            mean = self.df[column].mean()
            std = self.df[column].std()
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
            mask = (self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)
            self.df = self.df[mask]
        elif method == 'iqr':
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            mask = (self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)
            self.df = self.df[mask]
        
        return self.df
    
    def fit_distribution(self, column='PM2.5', distributions=None):
        """
        分布拟合（检验数据服从什么分布）
        
        Parameters:
        -----------
        column : str
            要拟合的列名
        distributions : list, optional
            要检验的分布列表，默认：['norm', 'gamma', 'lognorm']
        
        Returns:
        --------
        dict : 包含拟合结果的字典
        """
        if distributions is None:
            distributions = ['norm', 'gamma', 'lognorm']
        
        data = self.df[column].dropna()
        results = {}
        
        for dist_name in distributions:
            try:
                if dist_name == 'norm':
                    params = norm.fit(data)
                    ks_stat, p_value = stats.kstest(data, 'norm', args=params)
                    aic = self._calculate_aic(data, norm, params)
                    
                elif dist_name == 'gamma':
                    params = gamma.fit(data, floc=0)  # 固定位置参数为0
                    ks_stat, p_value = stats.kstest(data, 'gamma', args=params)
                    aic = self._calculate_aic(data, gamma, params)
                    
                elif dist_name == 'lognorm':
                    params = lognorm.fit(data, floc=0)
                    ks_stat, p_value = stats.kstest(data, 'lognorm', args=params)
                    aic = self._calculate_aic(data, lognorm, params)
                
                results[dist_name] = {
                    'params': params,
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'aic': aic
                }
            except Exception as e:
                results[dist_name] = {'error': str(e)}
        
        # 找出最佳分布（AIC最小）
        valid_results = {k: v for k, v in results.items() if 'aic' in v}
        if valid_results:
            best_dist = min(valid_results.items(), key=lambda x: x[1]['aic'])
            results['best_fit'] = best_dist[0]
        
        return results
    
    def _calculate_aic(self, data, dist, params):
        """计算AIC值"""
        try:
            log_likelihood = np.sum(dist.logpdf(data, *params))
            k = len(params)  # 参数个数
            n = len(data)  # 样本数
            aic = 2 * k - 2 * log_likelihood
            return aic
        except:
            return np.inf
    
    def log_transform(self, column='PM2.5', add_small_value=True):
        """
        Log变换：使数据正态化
        应用场景：PM2.5数据通常右偏，Log变换后可近似正态分布
        
        Parameters:
        -----------
        column : str
            要变换的列名
        add_small_value : bool
            是否在变换前添加小值（避免log(0)）
        
        Returns:
        --------
        pd.Series : 变换后的数据
        """
        if column not in self.df.columns:
            raise ValueError(f"列 {column} 不存在")
        
        data = self.df[column].copy()
        
        # 如果数据中有0或负值，添加小值
        if add_small_value and (data <= 0).any():
            data = data + 1e-6
        
        # Log变换
        log_data = np.log(data)
        
        return log_data
    
    def test_normality(self, data, test_type='shapiro'):
        """
        正态性检验
        
        Parameters:
        -----------
        data : array-like
            要检验的数据
        test_type : str
            检验类型：'shapiro'（Shapiro-Wilk）或 'normaltest'（D'Agostino）
        
        Returns:
        --------
        dict : 检验结果
        """
        data = np.array(data).flatten()
        data = data[~np.isnan(data)]
        
        if test_type == 'shapiro':
            # Shapiro-Wilk检验（适合小样本）
            if len(data) > 5000:
                # 如果样本太大，随机抽样
                data = np.random.choice(data, 5000, replace=False)
            stat, p_value = stats.shapiro(data)
            test_name = 'Shapiro-Wilk'
        elif test_type == 'normaltest':
            # D'Agostino's normality test
            stat, p_value = stats.normaltest(data)
            test_name = "D'Agostino"
        else:
            raise ValueError(f"不支持的检验类型: {test_type}")
        
        return {
            'test_name': test_name,
            'statistic': stat,
            'p_value': p_value,
            'is_normal': p_value > 0.05,
            'interpretation': f"{'服从' if p_value > 0.05 else '不服从'}正态分布 (p={p_value:.4f})"
        }
    
    def get_processed_data(self):
        """返回处理后的数据"""
        return self.df.copy()


if __name__ == "__main__":
    # 测试代码
    print("数据预处理模块已就绪")

