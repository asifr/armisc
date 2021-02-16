import numpy as np
from sklearn.metrics import (
    roc_curve,
    auc,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
)


def performance(y_test, yhat_test):
    """Returns the threshold, precision, recall, and specificty at
    the break-even point, Specificity@90% Specificity@95%."""
    fpr, tpr, roc_t = roc_curve(y_test, yhat_test)
    p, r, t = precision_recall_curve(y_test, yhat_test)

    breakeven_ind = np.argmax(np.minimum(p, r))
    breakeven_thresh = t[breakeven_ind]
    breakeven_precision = p[breakeven_ind]
    breakeven_recall = r[breakeven_ind]
    i_q = np.argmin(np.abs(roc_t - breakeven_thresh))
    breakeven_specificity = 1 - fpr[i_q]

    auc = roc_auc_score(y_test, yhat_test)

    spec90_ind = np.argmin(np.abs((1 - fpr) - 0.9))
    spec90_thresh = roc_t[spec90_ind]
    i_q = np.argmin(np.abs(t - spec90_thresh))
    spec90_recall = tpr[spec90_ind]
    spec90_precision = p[i_q]

    spec95_ind = np.argmin(np.abs((1 - fpr) - 0.95))
    spec95_thresh = roc_t[spec95_ind]
    i_q = np.argmin(np.abs(t - spec95_thresh))
    spec95_recall = tpr[spec95_ind]
    spec95_precision = p[i_q]

    ap = average_precision_score(y_test, yhat_test)
    auc_precision_recall = auc(r, p)
    prevalence = y_test.mean()

    return {
        "breakeven_thresh": breakeven_thresh,
        "breakeven_precision": breakeven_precision,
        "breakeven_recall": breakeven_recall,
        "breakeven_specificity": breakeven_specificity,
        "auc": auc,
        "auprc": auprc,
        "ap": ap,
        "spec90_thresh": spec90_thresh,
        "spec90_recall": spec90_recall,
        "spec90_precision": spec90_precision,
        "spec95_thresh": spec95_thresh,
        "spec95_recall": spec95_recall,
        "spec95_precision": spec95_precision,
        "prevalence": prevalence,
    }
