# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 20:53:57 2018

@author: Administrator
"""
# 导入pandas库，并将其简写为pd
import pandas as pd

# 读取Excel文件，并将数据存储在名为"data"的DataFrame对象中
data = pd.read_excel('data.xlsx')

# 选取data中指定的列，并创建一个新的DataFrame对象"data2"
data2 = data.iloc[:, [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]

# 删除"data2"中的缺失值
data2 = data2.dropna()

# 从sklearn.preprocessing模块导入StandardScaler类
from sklearn.preprocessing import StandardScaler

# 将"data2"中除第一列之外的数据赋值给变量"X"
X = data2.iloc[:, 1:]

# 创建一个StandardScaler对象
scaler = StandardScaler()

# 使用"X"来训练StandardScaler
scaler.fit(X)

# 使用训练好的StandardScaler将"X"标准化，并将结果赋值回"X"
X = scaler.transform(X)

# 从sklearn.decomposition模块导入PCA类
from sklearn.decomposition import PCA

# 创建一个PCA对象，保留累计贡献率达到95%的主成分
pca = PCA(n_components=0.95)

# 对已标准化的数据"X"进行PCA降维，并将结果赋值给"Y"
Y = pca.fit_transform(X)

# 计算各主成分的贡献率
gxl = pca.explained_variance_ratio_

# 导入numpy库，并将其简写为np
import numpy as np

# 创建一个长度为len(Y)的零向量F
F = np.zeros((len(Y)))

# 计算主成分的加权得分，并将结果累加到向量F
for i in range(len(gxl)):
    f = Y[:, i] * gxl[i]
    F = F + f

# 创建一个Series对象fs1，包含权重得分F和对应的股票代码
fs1 = pd.Series(F, index=data2['股票代码'].values)

# 对fs1中的得分进行降序排列
Fscore1 = fs1.sort_values(ascending=False)

# 读取另一个Excel文件，并将数据存储在名为"co"的DataFrame对象中
co = pd.read_excel('TRD_Co.xlsx')

# 创建一个Series对象Co，包含co中的股票名称和对应的股票代码
Co = pd.Series(co['Stknme'].values, index=co['Stkcd'].values)

# 从Co中选取与data2['股票代码']相对应的股票名称
Co1 = Co[data2['股票代码'].values]

# 创建一个Series对象fs2，包含权重得分F和对应的股票名称
fs2 = pd.Series(F, index=Co1.values)

# 对fs2中的得分进行降序排列
Fscore2 = fs2.sort_values(ascending=False)

# 打印按得分降序排列的股票代码和得分（Fscore1）
print(Fscore1)

# 打印按得分降序排列的股票名称和得分（Fscore2）
print(Fscore2)