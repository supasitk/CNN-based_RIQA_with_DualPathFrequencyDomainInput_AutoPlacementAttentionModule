from absl import logging
import sys
import operator
import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             roc_auc_score, roc_curve)


class accumulated_result(object):
    """Accumalate result of model's execution"""
    def __init__(self):
        self.accum_x_path = []
        self.accum_y_true = []
        self.accum_y_logit = []
        self.accum_y_prob = []
        self.accum_y_pred = []
        self.reset()
    
    def reset(self):
        self.accum_x_path.clear()
        self.accum_y_true.clear()
        self.accum_y_logit.clear()
        self.accum_y_prob.clear()
        self.accum_y_pred.clear()

    def update(self, x_path, y_true, y_logit):
        batch_size = y_logit.shape[0]
        y_prob = tf.nn.softmax(y_logit, axis=1)
        y_pred = np.argmax(y_logit, axis=1)
        self.accum_x_path.extend(np.reshape(x_path, [batch_size]))
        self.accum_y_true.extend(np.reshape(y_true, [batch_size]))
        for i in range(batch_size): # slow down execution!!!!!
            self.accum_y_logit.append(y_logit[i].numpy().tolist())
            self.accum_y_prob.append(y_prob[i].numpy().tolist())
        self.accum_y_pred.extend(np.reshape(y_pred, [batch_size]))


# [Start] code of Quadratic Weighted Kappa (QWK) metric from "https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/quadratic_weighted_kappa.py"
def qwk_confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Calculates the quadratic weighted kappa
    quadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.

    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.

    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.

    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = qwk_confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return 1.0 - numerator / denominator
# [End] code of Quadratic Weighted Kappa (QWK) metric from "https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/quadratic_weighted_kappa.py"


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

    # Quadratic Weighted Kappa (QWK) score
    first_class_id = 0
    last_class_id = n_class - 1
    QWK_score = quadratic_weighted_kappa(rater_a=y_true,
                                         rater_b=y_pred,
                                         min_rating=first_class_id,
                                         max_rating=last_class_id)

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

        'QWK': QWK_score,
    }

    return output


class record_top_N_validation_for_specific_metric(object):
  # dictionary for metrics information of validation
  MIN_FLOAT = sys.float_info.min
  MAX_FLOAT = sys.float_info.max

  val_infos_dict = {'Epoch': None,
                    'Loss': MAX_FLOAT,
                    'Accuracy': MIN_FLOAT,
                    'macroPrecision': MIN_FLOAT,
                    'macroSensitivity': MIN_FLOAT,
                    'macroF1': MIN_FLOAT,
                    'QWK': MIN_FLOAT
                   }


  def __init__(self, N, metric):
    # metric used for finding top-N validation
    self.metric = metric
    self.N = N

    # comparison operator of metric
    if self.metric == 'Loss': # if given loss is LESS than MAXIMUM (lowest rank) loss of top-N validation recored previously, 
                              # then appending this validation information to self.top_N_val_list, sorting it in ASCENDING [smallest to largest] order (lower loss is better), and finally getting only Top-N validation recorded to self.top_N_val_list
      self.comparison_op = operator.lt
      self.max_or_min_op = max
      self.sorted_reverse = False
    elif ((self.metric == 'Accuracy') or (self.metric == 'macroF1') or (self.metric == 'QWK')): # if given accuracy/macroF1/QWK is GREATER than MINIMUM (lowest rank) accuracy/macroF1/QWK of top-N validation recored previously,
                                                                                                # then appending this validation information to self.top_N_val_list, sorting it in DESCENDING [largest to smallest] order (greater accuracy/macroF1/QWK is better), and finally getting only Top-N validation recorded to self.top_N_val_list
      self.comparison_op = operator.gt
      self.max_or_min_op = min
      self.sorted_reverse = True 
    else:
      raise ValueError('Unknown \'{}\' metric of validation'.format(metric))

    # list of recording top-N validation (lower index is better validation [rank #1 to #N = index #0 to #N-1])
    self.top_N_val_list = [self.val_infos_dict for _ in range(N)]


  def update_top_N_validation(self, curr_val_info):
    curr_val_metric_value = curr_val_info[self.metric]
    lowestRankTopN_val_metric_value = self.max_or_min_op(self.top_N_val_list, key=lambda x: x[self.metric])[self.metric]
    self.is_current_val_metric_value_better = False
    
    if self.comparison_op(curr_val_metric_value, lowestRankTopN_val_metric_value):
      self.is_current_val_metric_value_better = True
      temp_top_N_val_list = self.top_N_val_list
      temp_top_N_val_list.append(curr_val_info)
      sorted_temp_top_N_val_list = sorted(temp_top_N_val_list, key=lambda x: x[self.metric], reverse=self.sorted_reverse)
      self.top_N_val_list = sorted_temp_top_N_val_list[:self.N]

  
  def show_top_N_validation(self):
    logging.info('\tTop-{} validation of {}:'.format(self.N, self.metric))
    for rank, val_info in enumerate(self.top_N_val_list):
      logging.info(str("\t\tRank #{}: ".format(rank+1)) +
                   str("Epoch {}".format(val_info['Epoch'])) + 
                   ', Loss: ' + str("{:0.4f}".format(val_info['Loss'])) +
                   ', Accuracy: ' + str("{:0.4f}".format(val_info['Accuracy'])) +
                   ', macroPrecision: ' + str("{:0.4f}".format(val_info['macroPrecision'])) +
                   ', macroSensitivity: ' + str("{:0.4f}".format(val_info['macroSensitivity'])) +
                   ', macroF1: ' + str("{:0.4f}".format(val_info['macroF1'])) +
                   ', QWK: ' + str("{:0.4f}".format(val_info['QWK'])))

  
  def get_epoch_number_in_top_N_validation(self):
    list_ep_num_in_topN_val = [val_info['Epoch'] for val_info in self.top_N_val_list if val_info['Epoch'] is not None]
    
    return list_ep_num_in_topN_val


class record_N_test_for_specific_val_metric(object):
  # dictionary for metrics information of test
  # test_infos_dict = {'Val_Rank': None,
  #                    'Loss': None,
  #                    'Accuracy': None,
  #                    'macroPrecision': None,
  #                    'macroSensitivity': None,
  #                    'macroF1': None,
  #                    'QWK': None
  #                   }


  def __init__(self, N, val_metric):
    # validation metric used for test
    self.val_metric = val_metric
    self.N = N
    
    # list of recording N test infos from top-N validation metric
    self.N_test_list = [None for _ in range(N)]
  

  def update_N_test(self, curr_test_info, rank):
    self.N_test_list[rank] = curr_test_info

  
  def show_best_worst_test(self):
    # sorting N test (list of dictionaries) first by 'Accuracy', 'QWK', and then by 'macroF1', 
    # and sorting it in DESCENDING (reverse=True) [largest to smallest] order (greater accuracy/macroF1/qwk is better)
    sorted_N_test_list = sorted(self.N_test_list, key=lambda x: (x['Accuracy'], x['QWK'], x['macroF1']), reverse=True)
    
    self.best_test = sorted_N_test_list[0]
    self.worst_test = sorted_N_test_list[self.N - 1]

    list_test = [('Best', self.best_test), ('Worst', self.worst_test)]

    for level, test in list_test:
      logging.info('{} Test of {} validation:'.format(level, self.val_metric))
      logging.info(str("\tFrom {} validation Rank #{} (epoch: {}):".format(self.val_metric, test['Val_Rank'], test['Val_Epoch'])) +
                   ' Loss: ' + str("{:0.4f}".format(test['Loss'])) +
                   ', Accuracy: ' + str("{:0.4f}".format(test['Accuracy'])) +
                   ', macroPrecision: ' + str("{:0.4f}".format(test['macroPrecision'])) +
                   ', macroSensitivity: ' + str("{:0.4f}".format(test['macroSensitivity'])) +
                   ', macroF1: ' + str("{:0.4f}".format(test['macroF1'])) + 
                   ', QWK: ' + str("{:0.4f}".format(test['QWK'])))