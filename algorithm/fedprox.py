from .fedbase import BasicServer, BasicClient
import copy
import torch
from utils import fmodule
import numpy as np
import utils.systemic_simulator as ss

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.algo_para = {'mu':0.1}
        self.init_algo_para(option['algo_para'])

    @ss.with_inactivity
    def sample(self):
        p = np.array(self.local_data_vols)[self.active_clients]
        p = p/np.sum(p)
        selected_clients = list(np.random.choice(self.active_clients, self.clients_per_round, replace=True, p=p))
        return selected_clients

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)

    @fmodule.with_multi_gpus
    def train(self, model):
        # global parameters
        src_model = copy.deepcopy(model)
        src_model.freeze_grad()
        model.train()
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.num_steps):
            # get a batch of data
            batch_data = self.get_batch_data()
            model.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            loss = self.calculator.train_one_step(model, batch_data)['loss']
            loss_proximal = 0
            for pm, ps in zip(model.parameters(), src_model.parameters()):
                loss_proximal += torch.sum(torch.pow(pm - ps, 2))
            loss = loss + 0.5 * self.mu * loss_proximal
            loss.backward()
            optimizer.step()
        return

