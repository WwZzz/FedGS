from .fedavg import Client
from .fedbase import BasicServer
import utils.systemic_simulator as ss
import numpy as np
from utils import fmodule

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)

    @ss.with_inactivity
    def sample(self):
        num_selected = min(self.clients_per_round, len(self.active_clients))
        selected_clients = list(np.random.choice(self.active_clients, num_selected, replace=False))
        return selected_clients

    def aggregate(self, models):
        datavols = np.array(self.local_data_vols)[self.selected_clients]
        p = datavols / self.total_data_vol
        sump = np.sum(p)
        p = p/sump
        return fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])
