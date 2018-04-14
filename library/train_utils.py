import numpy as np
from sklearn import metrics

def evaluate_prediction(y_true, y_pred, threshold=0.5):
	prec = calc_precision(y_true, y_pred)
	rec = calc_recall(y_true, y_pred)
	roc_auc = metrics.roc_auc_score(y, y_pred)
	return roc_auc, prec,  rec


def calc_precision(y_true, y_pred, th=0.5):
	pos_p = np.sum(np.round(y_pred - th + 0.5))
	if(pos_p == 0):
		return 1
	else:
		return np.sum(np.equal(np.sum(np.stack((np.round(y_pred - th + 0.5), y_true), axis=0), axis=0), np.full(len(y_true), 2))) / pos_p

def calc_recall(y_true, y_pred, th=0.5):
	return np.sum(np.equal(np.sum(np.stack((np.round(y_pred - th + 0.5), y_true), axis=0), axis=0), np.full(len(y_true), 2))) / np.sum(y_true)
