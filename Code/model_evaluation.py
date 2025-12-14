"""
模型评估模块
功能：统一计算回归和分类模型的评估指标，提供可视化功能
符合课程要求：模型评估与优化章节
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score
)
from statsmodels.stats.stattools import durbin_watson
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """统一模型评估类"""
    
    def __init__(self):
        """初始化评估器"""
        pass
    
    def regression_metrics(self, y_true, y_pred, model=None):
        """
        计算回归模型的评估指标
        
        Parameters:
        -----------
        y_true : array-like
            真实值
        y_pred : array-like
            预测值
        model : object, optional
            模型对象（用于计算AIC/BIC）
            
        Returns:
        --------
        dict : 包含所有回归评估指标的字典
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # 基本指标
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # 计算残差
        residuals = y_true - y_pred
        
        # AIC和BIC（如果提供了模型）
        aic = None
        bic = None
        if model is not None:
            try:
                if hasattr(model, 'aic'):
                    aic = model.aic
                if hasattr(model, 'bic'):
                    bic = model.bic
            except:
                pass
        
        # 如果没有从模型获取，尝试计算
        if aic is None or bic is None:
            try:
                n = len(y_true)
                k = 2  # 参数数量（简化估计）
                # 简化的AIC/BIC计算
                if aic is None:
                    aic = n * np.log(mse) + 2 * k
                if bic is None:
                    bic = n * np.log(mse) + k * np.log(n)
            except:
                pass
        
        result = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'aic': aic,
            'bic': bic,
            'residuals': residuals,
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals)
        }
        
        return result
    
    def classification_metrics(self, y_true, y_pred, y_proba=None):
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
            
        Returns:
        --------
        dict : 包含所有分类评估指标的字典
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
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
                y_proba = np.array(y_proba)
                if len(np.unique(y_true)) == 2:
                    # 二分类
                    auc_score = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    # 多分类：使用one-vs-rest策略
                    auc_score = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
            except Exception as e:
                auc_score = None
        
        result = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'confusion_matrix': cm,
            'auc_score': auc_score
        }
        
        return result
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, figsize=(8, 6)):
        """
        绘制混淆矩阵热力图
        
        Parameters:
        -----------
        y_true : array-like
            真实标签
        y_pred : array-like
            预测标签
        class_names : list, optional
            类别名称
        figsize : tuple
            图形大小
            
        Returns:
        --------
        fig, ax : matplotlib图形对象
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if class_names is None:
            n_classes = len(np.unique(y_true))
            class_names = [f'类别{i+1}' for i in range(n_classes)]
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )
        ax.set_xlabel('预测标签')
        ax.set_ylabel('真实标签')
        ax.set_title('混淆矩阵')
        
        return fig, ax
    
    def plot_roc_curve(self, y_true, y_proba, class_names=None, figsize=(8, 6)):
        """
        绘制ROC曲线（多分类使用one-vs-rest策略）
        
        Parameters:
        -----------
        y_true : array-like
            真实标签
        y_proba : array-like
            预测概率矩阵 (n_samples, n_classes)
        class_names : list, optional
            类别名称
        figsize : tuple
            图形大小
            
        Returns:
        --------
        fig, ax : matplotlib图形对象
        """
        y_true = np.array(y_true)
        y_proba = np.array(y_proba)
        
        n_classes = y_proba.shape[1]
        
        if class_names is None:
            class_names = [f'类别{i+1}' for i in range(n_classes)]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if n_classes == 2:
            # 二分类
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            auc = roc_auc_score(y_true, y_proba[:, 1])
            ax.plot(fpr, tpr, label=f'{class_names[1]} (AUC = {auc:.3f})')
        else:
            # 多分类：one-vs-rest
            for i in range(n_classes):
                y_binary = (y_true == i).astype(int)
                if len(np.unique(y_binary)) > 1:  # 确保有正负样本
                    fpr, tpr, _ = roc_curve(y_binary, y_proba[:, i])
                    auc = roc_auc_score(y_binary, y_proba[:, i])
                    ax.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', label='随机猜测')
        ax.set_xlabel('假正率 (FPR)')
        ax.set_ylabel('真正率 (TPR)')
        ax.set_title('ROC曲线')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    def plot_residuals(self, y_true, y_pred, figsize=(12, 4)):
        """
        绘制残差分析图
        
        Parameters:
        -----------
        y_true : array-like
            真实值
        y_pred : array-like
            预测值
        figsize : tuple
            图形大小
            
        Returns:
        --------
        fig : matplotlib图形对象
        """
        residuals = np.array(y_true) - np.array(y_pred)
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 1. 残差分布直方图
        axes[0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[0].axvline(0, color='r', linestyle='--', linewidth=2)
        axes[0].set_xlabel('残差')
        axes[0].set_ylabel('频数')
        axes[0].set_title('残差分布')
        axes[0].grid(True, alpha=0.3)
        
        # 2. 残差 vs 预测值
        axes[1].scatter(y_pred, residuals, alpha=0.5)
        axes[1].axhline(0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('预测值')
        axes[1].set_ylabel('残差')
        axes[1].set_title('残差 vs 预测值')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Q-Q图（残差正态性检验）
        stats.probplot(residuals, dist="norm", plot=axes[2])
        axes[2].set_title('Q-Q图（残差正态性）')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def durbin_watson_test(self, residuals):
        """
        Durbin-Watson检验（检验残差自相关性）
        
        Parameters:
        -----------
        residuals : array-like
            残差序列
            
        Returns:
        --------
        dict : 检验结果
        """
        residuals = np.array(residuals).flatten()
        
        try:
            dw_stat = durbin_watson(residuals)
            
            # DW统计量解释：
            # 0-2: 正自相关
            # 2: 无自相关
            # 2-4: 负自相关
            if dw_stat < 1.5:
                interpretation = "存在正自相关"
            elif dw_stat > 2.5:
                interpretation = "存在负自相关"
            else:
                interpretation = "无显著自相关"
            
            result = {
                'dw_statistic': dw_stat,
                'interpretation': interpretation,
                'is_independent': 1.5 < dw_stat < 2.5
            }
        except Exception as e:
            result = {
                'dw_statistic': None,
                'interpretation': f"计算失败: {str(e)}",
                'is_independent': None
            }
        
        return result
    
    def compare_models(self, models_dict, metric_type='regression'):
        """
        对比多个模型的性能
        
        Parameters:
        -----------
        models_dict : dict
            模型字典，格式：{'模型名': {'y_true': ..., 'y_pred': ..., 'y_proba': ...}}
        metric_type : str
            'regression' 或 'classification'
            
        Returns:
        --------
        pd.DataFrame : 对比表格
        """
        results = []
        
        for model_name, data in models_dict.items():
            if metric_type == 'regression':
                metrics = self.regression_metrics(
                    data['y_true'],
                    data['y_pred'],
                    model=data.get('model', None)
                )
                results.append({
                    '模型': model_name,
                    'RMSE': metrics['rmse'],
                    'MAE': metrics['mae'],
                    'R²': metrics['r2'],
                    'AIC': metrics['aic'],
                    'BIC': metrics['bic']
                })
            else:  # classification
                metrics = self.classification_metrics(
                    data['y_true'],
                    data['y_pred'],
                    y_proba=data.get('y_proba', None)
                )
                results.append({
                    '模型': model_name,
                    'Accuracy': metrics['accuracy'],
                    'Precision (Macro)': metrics['precision_macro'],
                    'Recall (Macro)': metrics['recall_macro'],
                    'F1-Score (Macro)': metrics['f1_macro'],
                    'AUC': metrics['auc_score']
                })
        
        df = pd.DataFrame(results)
        return df


if __name__ == "__main__":
    # 测试代码
    print("模型评估模块已就绪")
    print("功能：回归评估、分类评估、可视化、模型对比")

