import argparse
import importlib

def read_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', help='name of the benchmark;', type=str, default='mnist_classification')
    parser.add_argument('--dist', help='type of distribution;', type=int, default=0)
    parser.add_argument('--skew', help='the degree of niid;', type=float, default=0)
    parser.add_argument('--num_clients', help='the number of clients;', type=int, default=100)
    parser.add_argument('--seed', help='random seed;', type=int, default=0)
    try: option = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))
    return option

if __name__ == '__main__':
    option = read_option()
    TaskGen = getattr(importlib.import_module('.'.join(['benchmark', option['benchmark'], 'core'])), 'TaskGen')
    generator = TaskGen(dist_id = option['dist'], skewness = option['skew'], num_clients=option['num_clients'], seed = option['seed'])
    generator.run()
