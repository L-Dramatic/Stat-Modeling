"""
隐马尔可夫模型（HMM）模块
功能：将空气质量定义为隐状态，通过观测值反推污染等级
符合课程要求：HMM章节
"""

import pandas as pd
import numpy as np
from hmmlearn import hmm
import warnings
warnings.filterwarnings('ignore')


class HMMModel:
    """隐马尔可夫模型类"""
    
    def __init__(self, n_states=3):
        """
        初始化HMM模型
        
        Parameters:
        -----------
        n_states : int
            隐状态数量（例如：3 = 优良、轻度污染、重度污染）
        """
        self.n_states = n_states
        self.model = None
        self.state_names = None
        self.state_thresholds = None
    
    def _define_states(self, pm25_values):
        """
        定义隐状态（基于PM2.5值）
        
        Parameters:
        -----------
        pm25_values : array-like
            PM2.5值数组
        
        Returns:
        --------
        dict : 状态定义
        """
        # 根据中国空气质量标准定义状态
        # 优良: < 75, 轻度污染: 75-115, 重度污染: > 115
        thresholds = [0, 75, 115, np.inf]
        state_names = ['优良', '轻度污染', '重度污染']
        
        if self.n_states == 3:
            self.state_thresholds = thresholds
            self.state_names = state_names
        else:
            # 如果状态数不同，动态计算分位数
            percentiles = np.linspace(0, 100, self.n_states + 1)
            thresholds = np.percentile(pm25_values, percentiles)
            thresholds[0] = 0
            thresholds[-1] = np.inf
            self.state_thresholds = thresholds
            self.state_names = [f'状态{i+1}' for i in range(self.n_states)]
        
        return {
            'thresholds': self.state_thresholds,
            'names': self.state_names
        }
    
    def _pm25_to_state(self, pm25_value):
        """
        将PM2.5值转换为状态标签
        
        Parameters:
        -----------
        pm25_value : float
            PM2.5值
        
        Returns:
        --------
        int : 状态索引
        """
        for i, threshold in enumerate(self.state_thresholds[1:], 1):
            if pm25_value < threshold:
                return i - 1
        return self.n_states - 1
    
    def fit(self, observations, pm25_values=None):
        """
        拟合HMM模型
        
        Parameters:
        -----------
        observations : array-like, shape (n_samples, n_features)
            观测值（例如：温度、湿度、风速等）
        pm25_values : array-like, optional
            PM2.5值，用于定义状态
        """
        observations = np.array(observations)
        
        # 如果是1维，转换为2维
        if observations.ndim == 1:
            observations = observations.reshape(-1, 1)
        
        # 定义状态
        if pm25_values is not None:
            self._define_states(pm25_values)
        else:
            # 如果没有PM2.5值，使用默认定义
            self._define_states([0, 75, 115, 200])
        
        # 创建并拟合HMM模型
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=100
        )
        
        self.model.fit(observations)
        
        return self.model
    
    def predict_states(self, observations):
        """
        预测隐状态
        
        Parameters:
        -----------
        observations : array-like
            观测值
        
        Returns:
        --------
        np.array : 预测的状态序列
        """
        if self.model is None:
            raise ValueError("模型尚未拟合，请先调用fit()方法")
        
        observations = np.array(observations)
        if observations.ndim == 1:
            observations = observations.reshape(-1, 1)
        
        states = self.model.predict(observations)
        return states
    
    def get_transition_matrix(self):
        """
        获取状态转移矩阵
        
        Returns:
        --------
        pd.DataFrame : 状态转移矩阵
        """
        if self.model is None:
            raise ValueError("模型尚未拟合，请先调用fit()方法")
        
        trans_matrix = self.model.transmat_
        
        if self.state_names:
            df = pd.DataFrame(
                trans_matrix,
                index=self.state_names,
                columns=self.state_names
            )
        else:
            df = pd.DataFrame(trans_matrix)
        
        return df
    
    def get_emission_matrix(self):
        """
        获取发射概率矩阵（观测概率）
        
        Returns:
        --------
        dict : 发射概率信息
        """
        if self.model is None:
            raise ValueError("模型尚未拟合，请先调用fit()方法")
        
        # 对于高斯HMM，返回均值和协方差
        means = self.model.means_
        covars = self.model.covars_
        
        result = {
            'means': means,
            'covariances': covars,
            'state_names': self.state_names
        }
        
        return result
    
    def predict_current_state(self, current_observation):
        """
        预测当前隐状态（用于实时预警）
        
        Parameters:
        -----------
        current_observation : array-like
            当前观测值
        
        Returns:
        --------
        dict : 当前状态信息
        """
        if self.model is None:
            raise ValueError("模型尚未拟合，请先调用fit()方法")
        
        current_observation = np.array(current_observation)
        if current_observation.ndim == 1:
            current_observation = current_observation.reshape(1, -1)
        
        state = self.model.predict(current_observation)[0]
        state_prob = self.model.predict_proba(current_observation)[0]
        
        result = {
            'state_index': int(state),
            'state_name': self.state_names[state] if self.state_names else f'状态{state+1}',
            'state_probabilities': {
                self.state_names[i] if self.state_names else f'状态{i+1}': float(prob)
                for i, prob in enumerate(state_prob)
            },
            'most_likely_state': self.state_names[state] if self.state_names else f'状态{state+1}'
        }
        
        return result
    
    def get_model_summary(self):
        """
        获取模型摘要
        
        Returns:
        --------
        dict : 模型摘要信息
        """
        if self.model is None:
            raise ValueError("模型尚未拟合，请先调用fit()方法")
        
        return {
            'n_states': self.n_states,
            'state_names': self.state_names,
            'transition_matrix': self.get_transition_matrix(),
            'emission_info': self.get_emission_matrix(),
            'model_score': float(self.model.score(self.model.sample(100)[0]))
        }


if __name__ == "__main__":
    # 测试代码
    print("HMM模型模块已就绪")

