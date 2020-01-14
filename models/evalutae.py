import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
results = np.load('results.npy')
acc = accuracy_score(results[:,0], np.where(results[:,1]>=0.5, 1, 0))
auc = roc_auc_score(results[:,0], results[:,1])
print('Acc: {} AUC: {}'.format(acc, auc))