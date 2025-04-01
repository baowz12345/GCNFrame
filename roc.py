from GCNFrame import Biodata, GCNmodel
import torch
import numpy as np
from sklearn.metrics import accuracy_score,f1_score, matthews_corrcoef, confusion_matrix, roc_auc_score,auc
import matplotlib.pyplot as plt
#if __name__ == '__main__':
    # GCNmodel.test(model_name="GCN_model.pt", fasta_file="example_data/nature_2017.fasta")


true_labels = np.loadtxt('example_data/lifestyle_label.txt', dtype=int)
predicted_probs = np.loadtxt('test_output.txt',dtype=int)

pseudo_probs = np.where(predicted_probs == 1, 1.0, 0.0)

auroc = roc_auc_score(true_labels, predicted_probs)
from sklearn.metrics import roc_curve


fpr, tpr, thresholds = roc_curve(true_labels, pseudo_probs,drop_intermediate=False)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

print("AUROC:", auroc)