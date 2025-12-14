"""
分类模型模块
功能：实现Logistic Regression和Naive Bayes分类器，用于空气质量等级分类
符合课程要求：分类模型章节（硬性要求）
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings('ignore')


class ClassificationModels:
    """分类模型类"""
    
    def __init__(self):
        """初始化分类模型"""
        self.logistic_model = None
        self.naive_bayes_model = None
        self.scaler = StandardScaler()
        self.class_names = None
        self.is_fitted = False
    
    def _pm25_to_category(self, pm25_values):
        """
        将PM2.5连续值转换为分类标签
        
        Parameters:
        -----------
        pm25_values : array-like
            PM2.5浓度值
            
        Returns:
        --------
        np.array : 分类标签 (0: 优良, 1: 轻度污染, 2: 重度污染)
        """
        pm25_array = np.array(pm25_values)
        categories = np.zeros_like(pm25_array, dtype=int)
        
        # 根据中国空气质量标准
        categories[pm25_array >= 75] = 1  # 轻度污染
        categories[pm25_array >= 115] = 2  # 重度污染
        
        self.class_names = ['优良', '轻度污染', '重度污染']
        return categories
    
    def fit_logistic(self, X, y=None, pm25_values=None):
        """
        拟合多分类逻辑回归模型（Baseline模型）
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            特征变量（气象因子）
        y : pd.Series or np.array, optional
            分类标签（如果提供）
        pm25_values : array-like, optional
            PM2.5值（如果y未提供，将从此转换）
            
        Returns:
        --------
        LogisticRegression : 拟合后的模型
        """
        # 转换为数组
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.array(X)
        
        # 处理标签
        if y is None:
            if pm25_values is None:
                raise ValueError("必须提供y或pm25_values参数")
            y_array = self._pm25_to_category(pm25_values)
        else:
            y_array = np.array(y)
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X_array)
        
        # 拟合逻辑回归模型
        self.logistic_model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
        self.logistic_model.fit(X_scaled, y_array)
        self.is_fitted = True
        
        return self.logistic_model
    
    def fit_naive_bayes(self, X, y=None, pm25_values=None):
        """
        拟合朴素贝叶斯分类器（贝叶斯方法应用）
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            特征变量（气象因子）
        y : pd.Series or np.array, optional
            分类标签（如果提供）
        pm25_values : array-like, optional
            PM2.5值（如果y未提供，将从此转换）
            
        Returns:
        --------
        GaussianNB : 拟合后的模型
        """
        # 转换为数组
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.array(X)
        
        # 处理标签
        if y is None:
            if pm25_values is None:
                raise ValueError("必须提供y或pm25_values参数")
            y_array = self._pm25_to_category(pm25_values)
        else:
            y_array = np.array(y)
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X_array)
        
        # 拟合朴素贝叶斯模型
        self.naive_bayes_model = GaussianNB()
        self.naive_bayes_model.fit(X_scaled, y_array)
        self.is_fitted = True
        
        return self.naive_bayes_model
    
    def predict_logistic(self, X):
        """
        使用逻辑回归模型预测
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            特征变量
            
        Returns:
        --------
        np.array : 预测的类别
        """
        if self.logistic_model is None:
            raise ValueError("逻辑回归模型尚未拟合，请先调用fit_logistic()")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.array(X)
        
        X_scaled = self.scaler.transform(X_array)
        predictions = self.logistic_model.predict(X_scaled)
        
        return predictions
    
    def predict_naive_bayes(self, X):
        """
        使用朴素贝叶斯模型预测
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            特征变量
            
        Returns:
        --------
        np.array : 预测的类别
        """
        if self.naive_bayes_model is None:
            raise ValueError("朴素贝叶斯模型尚未拟合，请先调用fit_naive_bayes()")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.array(X)
        
        X_scaled = self.scaler.transform(X_array)
        predictions = self.naive_bayes_model.predict(X_scaled)
        
        return predictions
    
    def predict_proba_logistic(self, X):
        """
        使用逻辑回归模型预测概率
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            特征变量
            
        Returns:
        --------
        np.array : 预测概率矩阵 (n_samples, n_classes)
        """
        if self.logistic_model is None:
            raise ValueError("逻辑回归模型尚未拟合，请先调用fit_logistic()")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.array(X)
        
        X_scaled = self.scaler.transform(X_array)
        probabilities = self.logistic_model.predict_proba(X_scaled)
        
        return probabilities
    
    def predict_proba_naive_bayes(self, X):
        """
        使用朴素贝叶斯模型预测概率
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            特征变量
            
        Returns:
        --------
        np.array : 预测概率矩阵 (n_samples, n_classes)
        """
        if self.naive_bayes_model is None:
            raise ValueError("朴素贝叶斯模型尚未拟合，请先调用fit_naive_bayes()")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.array(X)
        
        X_scaled = self.scaler.transform(X_array)
        probabilities = self.naive_bayes_model.predict_proba(X_scaled)
        
        return probabilities
    
    def evaluate(self, y_true, y_pred, y_proba=None, model_name="Model"):
        """
        计算分类模型的评估指标
        
        Parameters:
        -----------
        y_true : array-like
            真实标签
        y_pred : array-like
            预测标签
        y_proba : array-like, optional
            预测概率（用于计算AUC）
        model_name : str
            模型名称
            
        Returns:
        --------
        dict : 包含所有评估指标的字典
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # 基本指标
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # 宏平均和微平均
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # AUC（多分类）
        auc_score = None
        if y_proba is not None:
            try:
                # 使用one-vs-rest策略计算多分类AUC
                if len(np.unique(y_true)) == 2:
                    auc_score = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    auc_score = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
            except:
                auc_score = None
        
        # 分类报告
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        
        result = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'confusion_matrix': cm,
            'auc_score': auc_score,
            'classification_report': report,
            'class_names': self.class_names
        }
        
        return result
    
    def get_class_names(self):
        """获取类别名称"""
        return self.class_names if self.class_names else ['优良', '轻度污染', '重度污染']


if __name__ == "__main__":
    # 测试代码
    print("分类模型模块已就绪")
    print("包含模型：Logistic Regression, Naive Bayes")

