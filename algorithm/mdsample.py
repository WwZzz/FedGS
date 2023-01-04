from .fedavg import Client
from .fedbase import BasicServer
import utils.systemic_simulator as ss
import numpy as np

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.client_sample_times = [0 for _ in range(self.num_clients)]

    @ss.with_inactivity
    def sample(self):
        p = np.array(self.local_data_vols)[self.active_clients]
        p = p/np.sum(p)
        selected_clients = list(np.random.choice(self.active_clients, self.clients_per_round, replace=True, p=p))
        return selected_clients