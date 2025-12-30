import copy
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import numpy as np
import time
from flcore.clients.clientbase import Client
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
from utils.privacy import *
from torchstat import stat


class clientGhp(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.round = 0
        self.args = args
        self.freeze = False
        self.TextEncoder = copy.deepcopy(args.TextEncoder)
        self.TextEncoder_opt = torch.optim.SGD(self.TextEncoder.parameters(), lr=self.learning_rate)
        self.TextEncoder_frozen = copy.deepcopy(self.TextEncoder)
        self.imaloss = nn.CrossEntropyLoss()
        self.txtloss = nn.CrossEntropyLoss()
        self.u = 1.0
        trainloader = self.load_train_data()
        self.sample_per_class = torch.zeros(self.num_classes).to(self.device)
        for x, y in trainloader:
            for yy in y:
                self.sample_per_class[yy.item()] += 1
        self.sample_per_class = self.sample_per_class / torch.sum(self.sample_per_class)


    def train(self, round):
        self.round = round
        trainloader = self.load_train_data()
        self.model.train()

        # differential privacy
        if self.privacy:
            self.model, self.optimizer, trainloader, privacy_engine = \
                initialize_dp(self.model, self.optimizer, trainloader, self.dp_sigma)

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for param in self.model.g_head.parameters():  # freeze param
            param.required_grad = False
        for param in self.TextEncoder_frozen.parameters():
            param.required_grad = False
        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                img_fea = self.model.base(x)
                text_fea_freeze = self.TextEncoder_frozen.embedding(torch.tensor(range(self.num_classes), device=self.device)).detach()#1,512,10
                text_fea = self.TextEncoder.embedding(torch.tensor(range(self.num_classes), device=self.device))
                img_fea_nor = img_fea / img_fea.norm(dim=1, keepdim=True)
                text_fea_freeze_nor = text_fea_freeze / text_fea_freeze.norm(dim=1, keepdim=True)
                text_fea_nor = text_fea / text_fea.norm(dim=1, keepdim=True)
                img_logits = 100 * img_fea_nor @ text_fea_freeze_nor.t()
                txt_logits = 100 * img_fea_nor.detach() @ text_fea_nor.t()

                loss = self.imaloss(img_logits, y)*self.u
                loss = loss+self.txtloss(txt_logits, y)*self.u

                embb = torch.zeros_like(text_fea[0])
                if self.freeze:
                    for l, emb in enumerate(text_fea_freeze):
                         embb += emb * self.sample_per_class[l]
                else:
                    for l, emb in enumerate(text_fea):
                        embb += emb * self.sample_per_class[l]
                embb = embb.unsqueeze(0).repeat(img_fea.size()[0],1)
                sup_img_fea = img_fea + embb.detach()
                out = self.model.head(sup_img_fea)+self.model.g_head(img_fea.detach()).detach()

                loss = loss + self.loss(out, y)
                self.optimizer.zero_grad()
                self.TextEncoder_opt.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.TextEncoder_opt.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")

    # def get_text_features_list(self, texts, model, device='cuda', train=False):
    #     if train:
    #         text_inputs = torch.cat([clip.tokenize(c) for c in texts]).to(device)
    #         text_features = model.encode_text(text_inputs)
    #     else:
    #         with torch.no_grad():
    #             text_inputs = torch.cat([clip.tokenize(c) for c in texts]).to(device)
    #             text_features = model.encode_text(text_inputs)
    #
    #     return text_features

    def test_metrics(self, model=None):
        testloader = self.load_test_data()
        if model == None:
            model = self.model
        model.eval()
        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                img_fea = self.model.base(x)
                text_fea_freeze = self.TextEncoder_frozen.embedding(torch.tensor(range(self.num_classes), device=self.device)).detach()
                text_fea = self.TextEncoder.embedding(torch.tensor(range(self.num_classes), device=self.device))

                embb = torch.zeros_like(text_fea[0])
                if self.freeze:
                    for l, emb in enumerate(text_fea_freeze):
                        embb += emb * self.sample_per_class[l]
                else:
                    for l, emb in enumerate(text_fea):
                        embb += emb * self.sample_per_class[l]

                embb = embb.unsqueeze(0).repeat(img_fea.size()[0], 1)
                sup_img_fea = img_fea + embb.detach()

                out = self.model.head(sup_img_fea)+self.model.g_head(img_fea.detach()).detach()
                test_acc += (torch.sum(torch.argmax(out, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(F.softmax(out).detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.eval()
        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                img_fea = self.model.base(x)
                text_fea_freeze = self.TextEncoder_frozen.embedding(torch.tensor(range(self.num_classes), device=self.device)).detach()#10,512
                text_fea = self.TextEncoder.embedding(torch.tensor(range(self.num_classes), device=self.device))
                img_fea_nor = img_fea / img_fea.norm(dim=1, keepdim=True)
                text_fea_freeze_nor = text_fea_freeze / text_fea_freeze.norm(dim=1, keepdim=True)
                text_fea = text_fea / text_fea.norm(dim=1, keepdim=True)
                img_logits = 100 * img_fea_nor @ text_fea_freeze_nor.t()
                txt_logits = 100 * img_fea_nor.detach() @ text_fea.t()

                loss = self.imaloss(img_logits, y)*self.u
                loss = loss+self.txtloss(txt_logits, y)*self.u

                embb = torch.zeros_like(text_fea[0])
                if self.freeze:
                    for l, emb in enumerate(text_fea_freeze):
                        embb += emb * self.sample_per_class[l]
                else:
                    for l, emb in enumerate(text_fea):
                        embb += emb * self.sample_per_class[l]

                embb = embb.unsqueeze(0).repeat(img_fea.size()[0], 1)
                sup_img_fea = img_fea + embb.detach()
                out = self.model.head(sup_img_fea)+self.model.g_head(img_fea.detach()).detach()
                loss = loss + self.loss(out, y)

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
        return losses, train_num


    def set_parameters(self, model):
        for new_param, old_param in zip(model.base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()
        for new_param, old_param in zip(model.head.parameters(), self.model.g_head.parameters()):
            old_param.data = new_param.data.clone()

    def set_TextEncoder(self, TextEncoder):
        for new_param, old_param in zip(TextEncoder.parameters(), self.TextEncoder.parameters()):
            old_param.data = new_param.data.clone()

        self.TextEncoder_frozen = copy.deepcopy(self.TextEncoder)


