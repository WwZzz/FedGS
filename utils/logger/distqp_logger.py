import utils.logger.basic_logger as bl
import torch
import numpy as np

class Logger(bl.Logger):
    def initialize(self, *args, **kwargs):
        for c in self.clients:
            data = torch.cat((c.train_data.X, c.valid_data.X)).mean(axis=0).tolist()
            self.output['client_optimal'].append(data)
        self.output['global_optimal'] = self.server.test_data.X.tolist()[0]

    def log_per_round(self):
        test_metric = self.server.test()
        for met_name, met_val in test_metric.items():
            self.output['test_' + met_name].append(met_val)
        # calculate weighted averaging of metrics on training datasets across clients
        train_metrics = self.server.test_on_clients('train')
        for met_name, met_val in train_metrics.items():
            self.output['train_' + met_name + '_dist'].append(met_val)
            self.output['train_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(self.server.local_data_vols, met_val)]) / self.server.total_data_vol)
        # calculate weighted averaging and other statistics of metrics on validation datasets across clients
        valid_metrics = self.server.test_on_clients('valid')
        for met_name, met_val in valid_metrics.items():
            self.output['valid_'+met_name+'_dist'].append(met_val)
            self.output['valid_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(self.server.local_data_vols, met_val)]) / self.server.total_data_vol)
            self.output['mean_valid_' + met_name].append(np.mean(met_val))
            self.output['std_valid_' + met_name].append(np.std(met_val))
        # 2D-trace for model
        self.output['model'].append(self.server.model.x.data.cpu().numpy().tolist()[0])
        # avtive_mode
        self.output['active_mode'].append([c.active for c in self.clients])
        self.output['selected_clients'].append(self.server.selected_clients)
        # output to stdout
        self.show_current_output()

    def organize_output(self):
        self.output['meta'] = self.meta
        for key in self.output.keys():
            if '_dist' in key:
                self.output[key] = self.output[key][-1]
        # data-size
        self.output['data_size'] = [len(c.train_data) for c in self.clients]
        # active_mode
        tmp = []
        for cid in range(len(self.clients)):
            for r in range(1, len(self.output['active_mode'])):
                if self.output['active_mode'][r][cid]:
                    tmp.append([r-1, cid])
        self.output['active_mode'] = tmp
        # sample_time
        self.output['sample_time'] = [0 for _ in self.clients]
        for r in range(len(self.output['selected_clients'])):
            for cid in self.output['selected_clients'][r]:
                self.output['sample_time'][cid] += 1
