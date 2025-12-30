import time
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing
from functools import partial
from flcore.clients.clientghp import clientGhp
from flcore.servers.serverbase import Server
from threading import Thread
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class FedGhp(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.feature_dim = list(args.model.head.parameters())[0].shape[1]
        args.TextEncoder = TextEncoder(in_features=self.feature_dim, num_classes=args.num_classes, dev=args.device).to(args.device)
        self.TextEncoder_sne = None
        # select slow clients
        self.args = args
        self.set_slow_clients()
        self.set_clients(clientGhp)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def train(self):
        round_protos = None
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()#defalt is all clients
            #self.send_models()#global model -> client model, cal send-time per round
            for client in self.selected_clients:
                client.train(i)

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
            # for client in self.selected_clients:
            #     client.train(i)

            self.receive_models()# cal every client weight, n/N
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()#aggregate clients para to global through the weights
            self.send_models()  # global model -> client model, cal send-time per round
            self.global_TextEncoder()



            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()


        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientGhp)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def global_TextEncoder(self):
        active_train_samples = 0
        for client in self.selected_clients:
            active_train_samples += client.train_samples

        self.uploaded_weights = []
        self.uploaded_model_gs = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples / active_train_samples)
            self.uploaded_model_gs.append(client.TextEncoder)

        self.TextEncoder = copy.deepcopy(self.uploaded_model_gs[0])
        for param in self.TextEncoder.parameters():
            param.data = torch.zeros_like(param.data)

        for w, client_model in zip(self.uploaded_weights, self.uploaded_model_gs):
            self.add_TextEncoder(w, client_model)

        for client in self.clients:
            client.set_TextEncoder(self.TextEncoder)

        self.TextEncoder_sne = copy.deepcopy(self.TextEncoder)

    def add_TextEncoder(self, w, TextEncoder):
        for server_param, client_param in zip(self.TextEncoder.parameters(), TextEncoder.parameters()):
            server_param.data += client_param.data.clone() * w


class TextEncoder(nn.Module):
    def __init__(self, in_features, num_classes, dev='cpu'):
        super(TextEncoder, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.embedding = nn.Embedding(num_classes, in_features)
        # self.conv1 = nn.Sequential(
        #     nn.Conv1d(3,512,1),
        #     nn.GELU()
        # )
        self.dev = dev

    def forward(self):
        embeddings = self.embedding(torch.tensor(range(self.num_classes), device=self.dev))
        # classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        # # a = ['airplane']
        # embeddings = clip.tokenize(classes)[:, :3].to(float).transpose(0, 1).unsqueeze(0)
        #embeddings = self.conv1(x)
        return embeddings

