from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from exp.exp_phy_loss import PhysicsRegularizedLoss # 此处的已经是第二版，实验效果更好的phy_loss了

# from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
# 将EarlyStopping修改为F2分数 -Tu
from utils.tools import adjust_learning_rate, cal_accuracy
from utils.myTools import EarlyStopping

import torch
import torch.nn as nn
import torch_optimizer as optim
import os
import time
import warnings
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, fbeta_score, average_precision_score

warnings.filterwarnings('ignore')


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _prepare_batch(self, batch):  # 编写预解包逻辑
        inputs, label = batch
        x_enc = inputs['x_enc'].float().to(self.device)
        text_emb = inputs['text_emb'].float().to(self.device) if inputs['text_emb'] is not None else None
        his_emb = inputs['his_emb'].float().to(self.device) if inputs['his_emb'] is not None else None
        raw_lc = inputs['raw_lc'].float().to(self.device)
        return x_enc, text_emb, his_emb, label.to(self.device), raw_lc

    def _build_model(self):
        self.args.seq_len = 512 # 这里的值需要不断更改来调参
        self.args.pred_len = 0 # 这里是0感觉不对，调试一下 ,分类任务中用不到，正确
        self.args.num_class = 2

        # 3. 加载模型
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss() #
        if self.args.on_phy_loss:
            criterion = PhysicsRegularizedLoss()
        return criterion

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        self.model.eval()
        preds = []
        trues = []

        with torch.no_grad():
            for batch in test_loader:
                x_enc, text_emb, his_emb, label, _ = self._prepare_batch(batch)
                outputs = self.model(x_enc, None, None, text_emb=text_emb, his_emb=his_emb)
                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        probs = torch.softmax(preds, dim=1)
        probs_positive = probs[:, 1].cpu().numpy()
        predictions = probs.argmax(dim=1).cpu().numpy()
        trues = trues.flatten().cpu().numpy()

        # --- 加权指标（整体）---
        acc_w = accuracy_score(trues, predictions)
        f1_w = f1_score(trues, predictions, average='weighted')
        rec_w = recall_score(trues, predictions, average='weighted')
        prec_w = precision_score(trues, predictions, average='weighted')

        # --- 正类指标（label=1）---
        acc = acc_w  # accuracy 无类别之分
        rec = recall_score(trues, predictions, pos_label=1, average='binary')
        prec = precision_score(trues, predictions, pos_label=1, average='binary')
        f1 = f1_score(trues, predictions, pos_label=1, average='binary')
        f2 = fbeta_score(trues, predictions, beta=2.0, pos_label=1, average='binary')
        auc_roc = roc_auc_score(trues, probs_positive)
        auc_pr = average_precision_score(trues, probs_positive)

        # 打印
        print("\n" + "=" * 50)
        print("【测试集结果】")
        print("=" * 50)
        print("【加权指标（整体性能）】")
        print(f"Accuracy: {acc_w:.6f}")
        print(f"F1 (weighted): {f1_w:.6f}")
        print(f"Recall (weighted): {rec_w:.6f}")
        print(f"Precision (weighted): {prec_w:.6f}")
        print("\n【正类指标（耀斑，label=1）】 ← 核心！")
        print(f"Recall (TPR): {rec:.6f}")
        print(f"Precision: {prec:.6f}")
        print(f"F1-score: {f1:.6f}")
        print(f"F2-score: {f2:.6f}")
        print(f"AUC-ROC: {auc_roc:.6f}")
        print(f"AUC-PR: {auc_pr:.6f}")
        print("=" * 50)

        # 保存
        folder_path = './results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)
        with open(os.path.join(folder_path, 'result_classification.txt'), 'a') as f:
            f.write(f"{setting}\n")
            f.write(f"Recall (positive): {rec:.6f}\n")
            f.write(f"Precision (positive): {prec:.6f}\n")
            f.write(f"F1-score (positive): {f1:.6f}\n")
            f.write(f"F2-score: {f2:.6f}\n")
            f.write(f"AUC-ROC: {auc_roc:.6f}\n")
            f.write(f"AUC-PR: {auc_pr:.6f}\n")
            f.write(f"Accuracy (weighted): {acc_w:.6f}\n")
            f.write("-" * 50 + "\n\n")

        return {
            'binary': {'recall': rec, 'precision': prec, 'f1': f1, 'f2': f2, 'auc_roc': auc_roc, 'auc_pr': auc_pr},
            'weighted': {'accuracy': acc_w, 'f1': f1_w, 'recall': rec_w, 'precision': prec_w}
        }

    def vali(self, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()

        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                x_enc, text_emb, his_emb, label, raw_lc = self._prepare_batch(batch)
                outputs = self.model(x_enc, None, None, text_emb=text_emb, his_emb=his_emb)  # (B, num_classes)

                # ✅ 正确计算 loss：在 device 上计算，label 需 squeeze 且为 long
                if self.args.on_phy_loss:
                    loss = criterion(outputs, label, raw_lc)
                else:
                    loss = criterion(outputs, label.squeeze().long())
                total_loss.append(loss.item())  # 只存标量

                preds.append(outputs.detach())
                trues.append(label.detach())

        # 合并所有 batch
        preds = torch.cat(preds, dim=0)  # (N, num_classes)
        trues = torch.cat(trues, dim=0)  # (N, 1)

        # 转为 numpy 用于 sklearn
        probs = torch.softmax(preds, dim=1)[:, 1].cpu().numpy()  # 正类概率
        trues_np = trues.flatten().cpu().numpy()  # (N,)

        # ✅ 计算正类 F2-score（β=2）—— 用于早停！
        predictions = (probs >= 0.5).astype(int)  # 默认阈值 0.5 #
        val_f1 = f1_score(trues_np, predictions, pos_label=1, average='binary')
        val_f2 = fbeta_score(trues_np, predictions, beta=2.0, pos_label=1, average='binary', zero_division=0)

        # # （可选）打印 F2 或 Recall 便于监控
        # val_recall = recall_score(trues_np, predictions, pos_label=1, average='binary', zero_division=0)
        # print(f"Val F2: {val_f2:.4f}, Recall: {val_recall:.4f}")
        #
        # self.model.train()
        # return np.average(total_loss), val_f2  # ← 返回 F2，不是 accuracy！

        # 早停指标改为f1
        val_recall = recall_score(trues_np, predictions, pos_label=1, average='binary', zero_division=0)
        print(f"Val F1: {val_f1:.4f}, Recall: {val_recall:.4f}")

        self.model.train()
        return np.average(total_loss), val_f1  # ← 返回 F1

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='VAL')
        # 注意：不再加载 test_loader 用于训练监控

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)

        # 使用 mode='max'，因为 F2 越大越好
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True,
                                       path=os.path.join(path, 'checkpoint.pth'), mode='max')

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()  # 假设是 nn.CrossEntropyLoss()

        for epoch in range(self.args.train_epochs):
            self.model.train()
            epoch_time = time.time()
            train_loss = []

            for i, batch in enumerate(train_loader):
                model_optim.zero_grad()

                x_enc, text_emb, his_emb, label, raw_lc = self._prepare_batch(batch)
                outputs = self.model(x_enc, None, None, text_emb=text_emb, his_emb=his_emb) # (base): (30,1,512)

                if self.args.on_phy_loss:
                    loss = criterion(outputs, label, raw_lc)
                else:
                    loss = criterion(outputs, label.squeeze().long())
                train_loss.append(loss.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 建议 1.0
                model_optim.step()

                if (i + 1) % 100 == 0:
                    iter_time = time.time() - time_now
                    speed = iter_time / (i + 1)
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")

            train_loss = np.average(train_loss)
            vali_loss, val_f1 = self.vali(vali_loader, criterion)  # 只用验证集

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.2f}s")
            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | "
                  f"Train Loss: {train_loss:.3f}, Vali Loss: {vali_loss:.3f}, Vali F2: {val_f1:.3f}")

            # 早停基于 F2-score（越大越好）
            early_stopping(val_f1, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # 加载最佳模型
        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model