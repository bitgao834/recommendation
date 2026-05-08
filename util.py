from sklearn.metrics import (
    jaccard_score,
    roc_auc_score,
    precision_score,
    f1_score,
    average_precision_score,
)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import random
import warnings
import dill
from collections import Counter
import torch

warnings.filterwarnings("ignore")

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def transform_split(X, Y):
    x_train, x_eval, y_train, y_eval = train_test_split(
        X, Y, train_size=2 / 3, random_state=1203
    )
    x_eval, x_test, y_eval, y_test = train_test_split(
        x_eval, y_eval, test_size=0.5, random_state=1203
    )
    return x_train, x_eval, x_test, y_train, y_eval, y_test

def sequence_output_process(output_logits, filter_token):
    pind = np.argsort(output_logits, axis=-1)[:, ::-1]
    out_list = []
    break_flag = False
    for i in range(len(pind)):
        if break_flag:
            break
        for j in range(pind.shape[1]):
            label = pind[i][j]
            if label in filter_token:
                break_flag = True
                break
            if label not in out_list:
                out_list.append(label)
                break
    y_pred_prob_tmp = [output_logits[idx, item] for idx, item in enumerate(out_list)]
    sorted_predict = [x for _, x in sorted(zip(y_pred_prob_tmp, out_list), reverse=True)]
    return out_list, sorted_predict

def sequence_metric(y_gt, y_pred, y_prob, y_label):
    def average_prc(y_gt, y_label):
        target = np.where(y_gt == 1)[0]
        inter = set(y_label) & set(target)
        return 0 if len(y_label) == 0 else len(inter) / len(y_label)

    def average_recall(y_gt, y_label):
        target = np.where(y_gt == 1)[0]
        inter = set(y_label) & set(target)
        return 0 if len(target) == 0 else len(inter) / len(target)

    def average_f1(average_prc, average_recall):
        return 0 if (average_prc + average_recall) == 0 else 2 * average_prc * average_recall / (average_prc + average_recall)

    def jaccard(y_gt, y_label):
        target = np.where(y_gt == 1)[0]
        inter = set(y_label) & set(target)
        union = set(y_label) | set(target)
        return 0 if len(union) == 0 else len(inter) / len(union)

    def f1(y_gt, y_pred):
        return f1_score(y_gt, y_pred, average='macro')

    def roc_auc(y_gt, y_pred_prob):
        return roc_auc_score(y_gt, y_pred_prob, average='macro')

    def precision_auc(y_gt, y_prob):
        return average_precision_score(y_gt, y_prob, average='macro')

    def precision_at_k(y_gt, y_prob_label, k):
        return sum(1 for j in y_prob_label[:k] if y_gt[j] == 1) / k

    try:
        auc = roc_auc(y_gt, y_prob)
    except ValueError:
        auc = 0

    p_1 = precision_at_k(y_gt, y_label, k=1)
    p_3 = precision_at_k(y_gt, y_label, k=3)
    p_5 = precision_at_k(y_gt, y_label, k=5)
    f1_val = f1(y_gt, y_pred)
    prauc = precision_auc(y_gt, y_prob)
    ja = jaccard(y_gt, y_label)
    avg_prc = average_prc(y_gt, y_label)
    avg_recall = average_recall(y_gt, y_label)
    avg_f1 = average_f1(avg_prc, avg_recall)

    return ja, prauc, avg_prc, avg_recall, avg_f1

def ddi_rate_score(record, path="../data/output/ddi_A_final.pkl"):
    with open(path, "rb") as f:
        ddi_A = dill.load(f)
    all_cnt = dd_cnt = 0
    for med_code_set in record:
        for i, med_i in enumerate(med_code_set):
            for j, med_j in enumerate(med_code_set):
                if j <= i:
                    continue
                all_cnt += 1
                if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                    dd_cnt += 1
    return 0 if all_cnt == 0 else dd_cnt / all_cnt


def pad_num_replace(tensor, old_val, new_val):
    return torch.where(tensor == old_val, torch.tensor(new_val, device=tensor.device), tensor)

def patient_to_visit(records, voc_size, max_history=None):
    visit_records = []
    for patient in records:
        for idx, adm in enumerate(patient):
            diag, proc, med = adm[0], adm[1], adm[2]

            start = 0
            if max_history is not None:
                start = max(0, idx - max_history)

            past = patient[start:idx]
            used_med  = [v[2] for v in past] if idx > 0 else []
            used_diag = [v[0] for v in past] if idx > 0 else []
            used_proc = [v[1] for v in past] if idx > 0 else []

            med_true = np.zeros(voc_size[2])
            med_true[med] = 1

            used_med_true = [np.zeros(voc_size[2]) for _ in used_med]
            for i, med_list in enumerate(used_med):
                used_med_true[i][med_list] = 1

            visit_records.append([diag, proc, med, used_med, used_diag, used_proc, med_true, used_med_true])
    return visit_records