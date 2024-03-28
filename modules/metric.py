import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             roc_auc_score, roc_curve)


class accumulated_result(object):
    """Accumalate result of model's execution"""
    def __init__(self):
        self.accum_x_path = []
        self.accum_y_true = []
#        self.accum_y_logit = []
        self.accum_y_prob = []
        self.accum_y_pred = []
        self.reset()
    
    def reset(self):
        self.accum_x_path.clear()
        self.accum_y_true.clear()
#        self.accum_y_logit.clear()
        self.accum_y_prob.clear()
        self.accum_y_pred.clear()

    def update(self, x_path, y_true, y_logit):
        batch_size = y_logit.shape[0]
        y_prob = tf.nn.softmax(y_logit, axis=1)
        y_pred = np.argmax(y_logit, axis=1)
        self.accum_x_path.extend(np.reshape(x_path, [batch_size]))
        self.accum_y_true.extend(np.reshape(y_true, [batch_size]))
        for i in range(batch_size): # slow down execution!!!!!
#            self.accum_y_logit.append(y_logit[i].numpy().tolist())
            self.accum_y_prob.append(y_prob[i].numpy().tolist())
        self.accum_y_pred.extend(np.reshape(y_pred, [batch_size]))


#def compute_metric(y_true, y_logit, y_prob, y_pred, class_names):
def compute_metric(y_true, y_prob, y_pred, class_names):
    """Evaluate meter for classification"""
    n_class = len(class_names)
    F1_metric = np.zeros([n_class, 1])
    tn = np.zeros([n_class, 1])
    fp = np.zeros([n_class, 1])
    fn = np.zeros([n_class, 1])
    tp = np.zeros([n_class, 1])

    Accuracy_score = accuracy_score(y_true, y_pred)
    ROC_curve = {}
    mAUC = 0

    for i in range(n_class):
        tmp_label = np.array(y_true) == i
        tmp_pred = np.array(y_pred) == i
        F1_metric[i] = f1_score(tmp_label, tmp_pred)
        tn[i], fp[i], fn[i], tp[i] = confusion_matrix(tmp_label, tmp_pred).ravel()
        outAUROC = roc_auc_score(tmp_label, np.array(y_prob)[:, i])

        mAUC = mAUC + outAUROC
        [roc_fpr, roc_tpr, roc_thresholds] = roc_curve(tmp_label, np.array(y_prob)[:, i])

        ROC_curve.update({'ROC_fpr_'+ str(i): roc_fpr,
                          'ROC_tpr_' + str(i): roc_tpr,
                          'ROC_T_' + str(i): roc_thresholds,
                          'AUC_' + str(i): outAUROC})
    
    mPrecision = sum(tp) / sum(tp + fp)
    mRecall = sum(tp) / sum(tp + fn)

    output = {
        'class_name': class_names,
        'F1': F1_metric,
        'AUC': mAUC / n_class,
        'Accuracy': Accuracy_score,

        'Sensitivity': tp / (tp + fn),
        'Precision': tp / (tp + fp),
        'Specificity': tn / (fp + tn),
        'ROC_curve': ROC_curve,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,

        'micro-Precision': mPrecision,
        'micro-Sensitivity': mRecall,
        'micro-Specificity': sum(tn) / sum(fp + tn),
        'micro-F1': 2*mPrecision * mRecall / (mPrecision + mRecall),
    }

    return output