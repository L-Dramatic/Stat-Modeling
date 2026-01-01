import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# 加载数据
df = pd.read_csv(r'c:\Users\29845\else\Desktop\Statistic\Stat-Modeling\Data\PRSA_data_2010.1.1-2014.12.31.csv')

# 处理缺失值
df = df.interpolate().ffill().bfill()

# 特征和目标
features = ['DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']
X = df[features].dropna()
y = df.loc[X.index, 'pm2.5'].dropna()
X = X.loc[y.index]

print(f"数据量: {len(X)} 条")
print()

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Lasso特征选择
print("正在进行Lasso特征选择（交叉验证）...")
lasso = LassoCV(cv=5, random_state=42, max_iter=2000)
lasso.fit(X_scaled, y)

# 输出结果
print()
print("="*60)
print("        Lasso 特征重要性分析结果")
print("="*60)
print(f"最优正则化参数 alpha: {lasso.alpha_:.6f}")
print()

results = []
for f, c in zip(features, lasso.coef_):
    results.append({
        'feature': f,
        'coef': c,
        'abs_coef': abs(c),
        'selected': '✅ 是' if abs(c) > 1e-5 else '❌ 否'
    })

results = sorted(results, key=lambda x: x['abs_coef'], reverse=True)

print(f"{'特征':<10} {'系数':>12} {'重要性(|系数|)':>15} {'是否选中':>10}")
print("-"*50)
for r in results:
    print(f"{r['feature']:<10} {r['coef']:>12.4f} {r['abs_coef']:>15.4f} {r['selected']:>10}")

selected_count = sum(1 for r in results if '是' in r['selected'])
print()
print(f"选中特征数量: {selected_count}/{len(features)}")
print()
print("="*60)
print("解读：")
print("- 系数为正：该因子增加时，PM2.5升高")
print("- 系数为负：该因子增加时，PM2.5降低")
print("- 系数=0：该特征被Lasso淘汰（不重要）")
print("="*60)
