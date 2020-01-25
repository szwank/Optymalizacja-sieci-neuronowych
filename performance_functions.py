from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

def accuracy_performance(answers, target):
    transformed_answers = np.max(answers)
    return accuracy_score(target, transformed_answers)

def AUC_performance(answers, target):
    return roc_auc_score(target, answers)
