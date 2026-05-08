import dill
import pickle
import random
import numpy as np
import argparse
from collections import defaultdict, Counter  # [修改] Counter 挪到顶部
from sklearn.metrics import jaccard_score
from torch.optim import Adam
import os
import sys
import torch
import time
from models import AIModel
from util import llprint, patient_to_visit, sequence_metric, ddi_rate_score, pad_num_replace
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from data_loader import mimic_data, pad_batch_v2_train, pad_batch_v2_eval
from sklearn.model_selection import train_test_split, KFold


# 控制台输出记录到文件
class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


torch.manual_seed(1203)
np.random.seed(2048)

# setting
model_name = "AIDrug"
resume_path = "./saved/AIDrug/Epoch_21_TARGET_0.06_JA_0.5463_DDI_0.06034.model"

if not os.path.exists(os.path.join("saved", model_name)):
    os.makedirs(os.path.join("saved", model_name))

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument("--Test", action="store_true", default=False, help="test mode")
parser.add_argument("--model_name", type=str, default=model_name, help="model name")
parser.add_argument("--resume_path", type=str, default=resume_path, help="resume path")
parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
parser.add_argument("--epoch", type=int, default=200, help="training epoches")
parser.add_argument("--target_ddi", type=float, default=0.06, help="target ddi")
parser.add_argument("--kgloss_alpha", type=float, default=0.5, help="kgloss_alpha for ddi_loss")
parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
parser.add_argument('--emb_dim', type=int, default=112, help='embedding dimension size')
parser.add_argument("--kp", type=float, default=0.05, help="coefficient of P signal")
parser.add_argument("--dim", type=int, default=64, help="dimension")
parser.add_argument("--cuda", type=int, default=0, help="which cuda")
parser.add_argument('--threshold', type=float, default=0.4, help='the threshold of prediction')

args = parser.parse_args()
print(vars(args))


def eval(model, eval_dataloader, ddi_adj, voc_size, device, TOKENS, args):
    model.eval()
    END_TOKEN, DIAG_PAD_TOKEN, PROC_PAD_TOKEN, MED_PAD_TOKEN, SOS_TOKEN = TOKENS
    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    med_cnt, visit_cnt = 0, 0

    for idx, data in enumerate(eval_dataloader):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        diagnoses, procedures, medications, used_medic, used_diag, used_proc, med_true, used_med_true, \
            d_mask_matrix, p_mask_matrix, m_mask_matrix = data

        diagnoses = pad_num_replace(diagnoses, -1, DIAG_PAD_TOKEN).to(device)
        procedures = pad_num_replace(procedures, -1, PROC_PAD_TOKEN).to(device)
        medications = medications.to(device)
        m_mask_matrix = m_mask_matrix.to(device)
        d_mask_matrix = d_mask_matrix.to(device)
        p_mask_matrix = p_mask_matrix.to(device)

        output_logits, ddi_loss = model(
            diagnoses, procedures, used_medic, used_diag, used_proc, used_med_true, d_mask_matrix, p_mask_matrix
        )

        visit_cnt += len(diagnoses)

        for i in range(len(diagnoses)):
            y_gt = med_true[i]
            current_pre = output_logits[i]
            prediction = torch.sigmoid(current_pre).cpu().detach().numpy()
            y_pred_prob = prediction
            out_list = np.where(prediction > args.threshold)[0]
            y_pred_label = out_list
            med_cnt += len(y_pred_label)
            y_pred = np.zeros(voc_size[2])
            y_pred[y_pred_label] = 1
            smm_record.append(y_pred_label)

            adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = \
                sequence_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob), np.array(y_pred_label))

            ja.append(adm_ja)
            prauc.append(adm_prauc)
            avg_p.append(adm_avg_p)
            avg_r.append(adm_avg_r)
            avg_f1.append(adm_avg_f1)

    ddi_rate = ddi_rate_score(smm_record, path="../data/output/ddi_A_final.pkl")

    llprint(
        "DDI Rate: {:.4}, Jaccard: {:.4}, PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n".format(
            ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt,
        )
    )

    return (
        ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p),
        np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt
    )


def main():
    log_path = "./log/"
    log_file_name = log_path + "log-" + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + ".log"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    sys.stdout = Logger(log_file_name)
    sys.stderr = Logger(log_file_name)
    print(vars(args))

    data_path = "../data/output/records_final.pkl"
    voc_path = "../data/output/voc_final.pkl"
    ddi_adj_path = "../data/output/ddi_A_final.pkl"
    ddi_mask_path = "../data/output/ddi_mask_H.pkl"
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    ddi_adj = dill.load(open(ddi_adj_path, "rb"))
    ddi_mask_H = dill.load(open(ddi_mask_path, "rb"))
    data_all = dill.load(open(data_path, "rb"))
    voc = dill.load(open(voc_path, "rb"))
    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
    END_TOKEN = voc_size[2] + 1
    DIAG_PAD_TOKEN = voc_size[0] + 2
    PROC_PAD_TOKEN = voc_size[1] + 2
    MED_PAD_TOKEN = voc_size[2] + 2
    SOS_TOKEN = voc_size[2]
    TOKENS = [END_TOKEN, DIAG_PAD_TOKEN, PROC_PAD_TOKEN, MED_PAD_TOKEN, SOS_TOKEN]

    fold_settings = [5]
    all_data = data_all
    results_summary = {}

    for n_fold in fold_settings:
        print(f"\n===== 开始 {n_fold}-折交叉验证 =====")
        kf = KFold(n_splits=n_fold, shuffle=True, random_state=42)
        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(all_data)):
            print(f"\n----- Fold {fold + 1} / {n_fold} -----")
            data_train = [all_data[i] for i in train_idx]
            data_test_full = [all_data[i] for i in test_idx]
            val_split = len(data_test_full) // 2
            data_test = data_test_full[:val_split]
            data_eval = data_test_full[val_split:]

            data_train_v = patient_to_visit(data_train, voc_size, max_history=10)


            # ===== 全量统计 history length（用于判断是否存在长历史）=====
            hist_lens = [len(v[3]) for v in data_train_v]  # v[3] == used_med


            data_test_v = patient_to_visit(data_test, voc_size)
            data_eval_v = patient_to_visit(data_eval, voc_size)

            train_dataset = mimic_data(data_train_v)
            eval_dataset = mimic_data(data_eval_v)
            test_dataset = mimic_data(data_test_v)

            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                          collate_fn=pad_batch_v2_train, shuffle=True, pin_memory=True)
            eval_dataloader = DataLoader(eval_dataset, batch_size=16,
                                         collate_fn=pad_batch_v2_eval, shuffle=False, pin_memory=True)

            model = AIModel(voc_size, ddi_adj, emb_dim=args.emb_dim, kgloss_alpha=args.kgloss_alpha, device=device)
            model.to(device)
            optimizer = Adam(model.parameters(), lr=args.lr)



            best_ja = 0
            epoch_metrics = {'loss': [], 'bce': [], 'ddi': [], 'ja': [], 'f1': [], 'prauc': [], 'avg_med': [],
                             'ddi_rate': []}

            for epoch in range(args.epoch):
                model.train()
                epoch_loss = epoch_loss_current = epoch_loss_ddi = batch_count = 0

                for idx, data in enumerate(train_dataloader):
                    diagnoses, procedures, medications, used_medic, used_diag, used_proc, med_true, used_med_true, \
                        d_mask_matrix, p_mask_matrix, m_mask_matrix = data


                    diagnoses = pad_num_replace(diagnoses, -1, DIAG_PAD_TOKEN).to(device)
                    procedures = pad_num_replace(procedures, -1, PROC_PAD_TOKEN).to(device)
                    medications = medications.to(device)
                    d_mask_matrix = d_mask_matrix.to(device)
                    p_mask_matrix = p_mask_matrix.to(device)
                    m_mask_matrix = m_mask_matrix.to(device)

                    output1, loss_ddi = model(
                        diagnoses, procedures, used_medic, used_diag, used_proc, used_med_true, d_mask_matrix,
                        p_mask_matrix
                    )

                    loss_current = F.binary_cross_entropy_with_logits(output1, torch.tensor(med_true).to(device))
                    loss = loss_current + loss_ddi
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    epoch_loss_current += loss_current.item()
                    epoch_loss_ddi += loss_ddi.item()
                    batch_count += 1

                print(
                    f"Epoch {epoch}: 总损失={epoch_loss / batch_count:.4f}, BCE={epoch_loss_current / batch_count:.4f}, DDI={epoch_loss_ddi / batch_count:.4f}"
                )

                ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, eval_dataloader, ddi_adj, voc_size,
                                                                          device, TOKENS, args)

                epoch_metrics['loss'].append(epoch_loss / batch_count)
                epoch_metrics['bce'].append(epoch_loss_current / batch_count)
                epoch_metrics['ddi'].append(epoch_loss_ddi / batch_count)
                epoch_metrics['ja'].append(ja)
                epoch_metrics['f1'].append(avg_f1)
                epoch_metrics['prauc'].append(prauc)
                epoch_metrics['avg_med'].append(avg_med)
                epoch_metrics['ddi_rate'].append(ddi_rate)
                if ja > best_ja:
                    best_ja = ja

                if (epoch + 1) % 10 == 0:
                    start_ep = epoch - 9
                    end_ep = epoch
                    print(f"\nEpoch {start_ep}-{end_ep}: "
                          f"DDI Rate: {np.mean(epoch_metrics['ddi_rate'][-10:]):.4f} ± {np.std(epoch_metrics['ddi_rate'][-10:]):.4f}\n"
                          f"Jaccard: {np.mean(epoch_metrics['ja'][-10:]):.4f} ± {np.std(epoch_metrics['ja'][-10:]):.4f}\n  "
                          f"F1: {np.mean(epoch_metrics['f1'][-10:]):.4f} ± {np.std(epoch_metrics['f1'][-10:]):.4f}\n  "
                          f"PRAUC: {np.mean(epoch_metrics['prauc'][-10:]):.4f} ± {np.std(epoch_metrics['prauc'][-10:]):.4f}\n  "
                          f"AVG_MED: {np.mean(epoch_metrics['avg_med'][-10:]):.4f} ± {np.std(epoch_metrics['avg_med'][-10:]):.4f}\n"
                          )

            fold_results.append(epoch_metrics)
        results_summary[n_fold] = fold_results

    print("\n===== 五折交叉验证完成 =====")
    for n_fold in fold_settings:
        print(f"\n===== {n_fold}-折交叉验证结果汇总 =====")
        for metric in ['ja', 'f1', 'ddi', 'prauc', 'avg_med']:
            scores = [np.mean(fold[metric]) for fold in results_summary[n_fold]]
            print(f"{metric}: Mean={np.mean(scores):.4f}, Std={np.std(scores):.4f}")


if __name__ == "__main__":
    main()