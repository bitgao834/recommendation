import torch
import torch.nn as nn
import numpy as np


class AIModel(nn.Module):
    """
    Ablation: w/o D/P Transformer

    说明：
    - 当前 visit：D/P 不再经过 TransformerEncoderLayer，
      而是直接 Embedding 后做 sum pooling
    - 上一次用药（last meds）：保留
    - 历史处方序列（m_h）：保留，仍然使用 used_med_true -> GRU
    - 最终 logits = 当前分支 logits + 历史GRU logits
    - DDI loss 保持不变

    这样做的目的：
    验证 D/P Transformer 的 visit 内部依赖建模是否有效
    """

    def __init__(self, voc_size, ddi_adj, emb_dim, kgloss_alpha, device=torch.device("cpu:0")):
        super(AIModel, self).__init__()
        self.device = device
        self.ddi_adj = torch.FloatTensor(ddi_adj).to(self.device)

        self.DIAG_PAD_TOKEN = voc_size[0] + 2
        self.PROC_PAD_TOKEN = voc_size[1] + 2
        self.MED_PAD_TOKEN = voc_size[2] + 2

        self.med_vocab_size = voc_size[2]
        self.emb_dim = emb_dim
        self.kgloss_alpha = kgloss_alpha

        # Embedding
        self.diag_embedding = nn.Sequential(
            nn.Embedding(voc_size[0] + 3, emb_dim, padding_idx=self.DIAG_PAD_TOKEN),
            nn.Dropout(0.3)
        )
        self.proc_embedding = nn.Sequential(
            nn.Embedding(voc_size[1] + 3, emb_dim, padding_idx=self.PROC_PAD_TOKEN),
            nn.Dropout(0.3)
        )

        # last meds embedding
        self.used_med_embedding = nn.Sequential(
            nn.Embedding(voc_size[2] + 3, emb_dim, padding_idx=self.MED_PAD_TOKEN),
            nn.Dropout(0.3)
        )

        # 当前 visit 输出层：输入 3*emb_dim -> 输出 med_vocab logits
        self.output_layer = nn.Linear(emb_dim * 3, voc_size[2])

        # 历史处方序列 GRU（m_h）：input_size=med_vocab (multi-hot)，hidden_size=med_vocab
        self.encoders = nn.GRU(
            input_size=voc_size[2],
            hidden_size=voc_size[2],
            num_layers=1,
            batch_first=True
        )

    def used_med_learning_from_true(self, used_med_true):
        """
        used_med_true: list，长度 batch_size
          used_med_true[i] = [vec0, vec1, ..., vec(T-1)]
          其中 veck 是 np.ndarray 或 list，shape = [med_vocab]，multi-hot(0/1)

        输出:
          history_logits: Tensor [B, med_vocab]
        """
        batch_size = len(used_med_true)
        med_voc = self.med_vocab_size
        history_logits = torch.zeros(batch_size, med_voc, device=self.device)

        for i in range(batch_size):
            hist_seq = used_med_true[i]

            if hist_seq is None or len(hist_seq) == 0:
                continue

            seq = torch.tensor(np.array(hist_seq), dtype=torch.float32, device=self.device)
            if seq.dim() != 2 or seq.size(1) != med_voc:
                raise ValueError(
                    f"used_med_true[{i}] has invalid shape: {tuple(seq.shape)}, expected [T, {med_voc}]"
                )

            _, h_n = self.encoders(seq.unsqueeze(0))  # [1,T,V] -> h_n [1,1,V]
            history_logits[i] = h_n[-1, 0, :]

        return history_logits

    def forward(
        self,
        diagnose,
        procedures,
        used_medications,
        used_diag,
        used_proc,
        used_med_true,
        d_mask_matrix,
        p_mask_matrix
    ):
        """
        参数说明（与 main.py / data_loader.py 输出一致）：
        - diagnose: Tensor [B, D]
        - procedures: Tensor [B, P]
        - used_medications: list（仅用于取 last meds）
        - used_med_true: list（历史 multi-hot 序列，用于 GRU）
        - d_mask_matrix/p_mask_matrix: 为了兼容原训练代码而保留，但本消融中不再使用
        """
        batch_size = diagnose.shape[0]

        # 1) 当前 visit 的 D/P 表示：不经过 Transformer，直接 embedding + sum pooling
        diag_emb = self.diag_embedding(diagnose)      # [B, D, emb]
        proc_emb = self.proc_embedding(procedures)    # [B, P, emb]

        diag_pool = torch.sum(diag_emb, dim=1)        # [B, emb]
        proc_pool = torch.sum(proc_emb, dim=1)        # [B, emb]

        # 2) last meds embedding -> [B, emb]
        last_med_pool = []
        for i in range(batch_size):
            if used_medications is not None and len(used_medications[i]) != 0:
                last_visit_meds = used_medications[i][-1]
                if isinstance(last_visit_meds, (list, tuple, np.ndarray)) and len(last_visit_meds) > 0:
                    med_ids = torch.tensor(last_visit_meds, device=self.device, dtype=torch.long)
                    med_emb = self.used_med_embedding(med_ids).sum(dim=0)  # [emb]
                else:
                    med_emb = torch.zeros(self.emb_dim, device=self.device)
            else:
                med_emb = torch.zeros(self.emb_dim, device=self.device)

            last_med_pool.append(med_emb)

        last_med_pool = torch.stack(last_med_pool, dim=0)  # [B, emb]

        # 3) 当前分支 logits
        final_input = torch.cat([diag_pool, proc_pool, last_med_pool], dim=1)  # [B, 3*emb]
        decoder_output = self.output_layer(final_input)  # [B, med_vocab]

        # 4) 历史 GRU logits
        history_logits = self.used_med_learning_from_true(used_med_true)  # [B, med_vocab]
        decoder_output = decoder_output + history_logits

        # 5) DDI loss
        sigmoid_output = torch.sigmoid(decoder_output)  # [B, V]
        sigmoid_output_ddi = torch.matmul(
            sigmoid_output.unsqueeze(2), sigmoid_output.unsqueeze(1)
        )  # [B, V, V]

        kg_ddi = self.ddi_adj.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, V, V]
        kg_ddi_score = 0.001 * self.kgloss_alpha * torch.sum(
            kg_ddi * sigmoid_output_ddi, dim=[1, 2]
        ).mean()

        return decoder_output, kg_ddi_score