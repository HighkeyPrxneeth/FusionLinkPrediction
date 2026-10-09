import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def metrics(labels, scores):
    y, s = np.asarray(labels), np.asarray(scores)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {'auc': None, 'ap': None, 'count': len(y), 'positives': int(y.sum()), 'prevalence': float(y.mean()) if len(y) else None}
    return {'auc': float(roc_auc_score(y, s)), 'ap': float(average_precision_score(y, s)), 'count': len(y), 'positives': int(y.sum()), 'prevalence': float(y.mean())}


def fit_predict(fusion_features, fusion_labels, test_features, c=1.0):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000, solver='lbfgs'))
    model.fit(fusion_features, fusion_labels)
    score = model.predict_proba(test_features)[:, 1]
    return score, model
