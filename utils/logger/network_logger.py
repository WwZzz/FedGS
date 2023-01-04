import utils.logger.basic_logger as bl
import numpy as np
import collections

class Logger(bl.Logger):
    def initialize(self, *args, **kwargs):
        self.output['sampled_times'] = [0 for _ in range(len(self.clients))]

    def log_per_round(self, *args, **kwargs):
        # active mode
        self.output['active_mode'].append([c.active for c in self.clients])
        tmp = [int(sid) for sid in self.server.selected_clients]
        self.output['selected_clients'].append(tmp)
        for cid in self.server.selected_clients:
            self.output['sampled_times'][cid] += 1
        self.show_current_output()

    def organize_output(self, *args, **kwargs):
        self.output['meta'] = self.meta
        num_samples = [0 for _ in self.clients]
        for r in self.output['selected_clients']:
            for cid in r:
                num_samples[cid]+=1
        self.output['sampled_time_dist'] = num_samples
        # sort active mode for clients
        a = np.array(self.output['active_mode'], dtype=int)
        a = a[1:]
        sa = np.sum(a.T, axis=1)
        sorted_clients = sa.argsort().tolist()
        idx = {s:i for s,i in zip(sorted_clients, [i for i in range(len(self.clients))])}
        # active_mode
        tmp = []
        for cid in range(len(self.clients)):
            for r in range(1, len(self.output['active_mode'])):
                if self.output['active_mode'][r][cid]:
                    tmp.append([r - 1, idx[cid]])
        self.output['active_mode'] = tmp
        # selected_clients
        selected_clients = []
        for r in range(1, len(self.output['selected_clients'])):
            for cid in self.output['selected_clients'][r]:
                selected_clients.append([r-1, idx[cid]])
        self.output['selected_clients'] = selected_clients
        return
