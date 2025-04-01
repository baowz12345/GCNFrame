import numpy as np
from sklearn.metrics import accuracy_score,f1_score, matthews_corrcoef, confusion_matrix, roc_auc_score
from GCNFrame import Biodata, GCNmodel
import torch

# 从.txt文件加载真实标签和预测标签，并将它们转换为NumPy数组
true_labels = np.loadtxt('example_data/lifestyle_label.txt', dtype=int)
predicted_labels = np.loadtxt('test_output.txt', dtype=int)

# 计算准确率
accuracy = accuracy_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

# 计算MCC（马修斯相关系数）
mcc = matthews_corrcoef(true_labels, predicted_labels)

# 计算混淆矩阵，以计算敏感性和特异性
conf_matrix = confusion_matrix(true_labels, predicted_labels)
tn, fp, fn, tp = conf_matrix.ravel()

# 计算敏感性和特异性
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print(f"准确率：{accuracy * 100:.3f}%")
print(f"F1分数：{f1* 100:.3f}%")
print(f"MCC（马修斯相关系数）：{mcc* 100:.3f}%")
print(f"敏感性：{sensitivity* 100:.3f}%")
print(f"特异性：{specificity* 100:.3f}%")
