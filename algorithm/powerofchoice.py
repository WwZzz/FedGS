import numpy as np
from .fedavg import Client
from .fedbase import BasicServer
import copy
import utils.systemic_simulator as ss

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)

    @ss.with_inactivity
    def sample(self):
        losses = []
        for cid in self.active_clients:
            losses.append(self.clients[cid].test(self.model)['loss'])
        sort_id = np.array(losses).argsort().tolist()
        sort_id.reverse()
        num_selected = min(self.clients_per_round, len(self.active_clients))
        selected_clients = np.array(self.active_clients)[sort_id][:num_selected]
        return selected_clients.tolist()

    def iterate(self, t):
        self.selected_clients = self.sample()
        # training
        models = self.communicate(self.selected_clients)['model']
        # aggregate: pk = 1/K as default where K=len(selected_clients)
        self.model = self.aggregate(models)
        return
