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


class record_top_N_validation_for_specific_metric(object):
  # dictionary for metrics information of validation
  MIN_FLOAT = sys.float_info.min
  MAX_FLOAT = sys.float_info.max

  val_infos_dict = {'Epoch': None,
                    'Loss': MAX_FLOAT,
                    'Accuracy': MIN_FLOAT,
                    'macroPrecision': MIN_FLOAT,
                    'macroSensitivity': MIN_FLOAT,
                    'macroF1': MIN_FLOAT
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
    elif ((self.metric == 'Accuracy') or (self.metric == 'macroF1')): # if given accuracy/macroF1 is GREATER than MINIMUM (lowest rank) accuracy/macroF1 of top-N validation recored previously,
                                                                      # then appending this validation information to self.top_N_val_list, sorting it in DESCENDING [largest to smallest] order (greater accuracy is better), and finally getting only Top-N validation recorded to self.top_N_val_list
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
                   ', macroF1: ' + str("{:0.4f}".format(val_info['macroF1'])))

  
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
  #                    'macroF1': None
  #                   }


  def __init__(self, N, val_metric):
    # validation metric used for test
    self.val_metric = val_metric
    self.N = N
    
    # list of recording N test infos from top-N validation metric
    self.N_test_list = [None for _ in range(N)]
  

  def update_N_test(self, curr_test_info, rank):
    self.N_test_list[rank] = curr_test_info

  
  def show_best_test(self):
    # sorting N test (list of dictionaries) first by 'Accuracy' and then by 'macroF1', 
    # and sorting it in DESCENDING (reverse=True) [largest to smallest] order (greater accuracy/macroF1 is better)
    sorted_N_test_list = sorted(self.N_test_list, key=lambda x: (x['Accuracy'], x['macroF1']), reverse=True)
    
    self.best_test = sorted_N_test_list[0]

    logging.info(str("\tFrom {} validation Rank #{}:".format(self.val_metric, self.best_test['Val_Rank'])) +
                 ' Loss: ' + str("{:0.4f}".format(self.best_test['Loss'])) +
                 ', Accuracy: ' + str("{:0.4f}".format(self.best_test['Accuracy'])) +
                 ', macroPrecision: ' + str("{:0.4f}".format(self.best_test['macroPrecision'])) +
                 ', macroSensitivity: ' + str("{:0.4f}".format(self.best_test['macroSensitivity'])) +
                 ', macroF1: ' + str("{:0.4f}".format(self.best_test['macroF1'])))