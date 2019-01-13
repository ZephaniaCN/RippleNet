
from train import train
import threading

import argparse
import numpy as np
import random
from data_loader  import load_data
from hyperopt import hp,Trials,fmin,tpe
import csv


np.random.seed(589)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
parser.add_argument('--model',type=str,default='ripple_net_plus', help='which model to use')
parser.add_argument('--dim', type=int, default=16, help='dimension of entity and relation embeddings')
parser.add_argument('--n_hop', type=int, default=4, help='maximum hops origin:2')
parser.add_argument('--kge_weight', type=float, default=0.01, help='weight of the KGE term')
parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of the l2 regularization term')
parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--n_epoch', type=int, default=15, help='the number of epochs, origin:10')
parser.add_argument('--n_memory', type=int, default=32, help='size of ripple set for each hop')
parser.add_argument('--embed_size', type=int, default=16,
                    help=' the number of output units in the first layer of attention c')
parser.add_argument('--mongo', type=str,default='haha',
                    help=' the number of output units in the first layer of attention c')
parser.add_argument('--poll-interval', type=int,default=16)
parser.add_argument('--max-jobs', type=int,default=16)
parser.add_argument('--reserve-timeout', type=float,default=16)
# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
# parser.add_argument('--dim', type=int, default=16, help='dimension of entity and relation embeddings')
# parser.add_argument('--n_hop', type=int, default=2, help='maximum hops origin:2')
# parser.add_argument('--kge_weight', type=float, default=0.01, help='weight of the KGE term')
# parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of the l2 regularization term')
# parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
# parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
# parser.add_argument('--n_epoch', type=int, default=10, help='the number of epochs, origin:10')
# parser.add_argument('--n_memory', type=int, default=32, help='size of ripple set for each hop')
# parser.add_argument('--item_update_mode', type=str, default='plus_transform',
#                     help='how to update item at the end of each hop')
# parser.add_argument('--using_all_hops', type=bool, default=True,
#                     help='whether using outputs of all hops or just the last hop when making prediction')
# parser.add_argument('--embed_size', type=bool, default=True,
#                     help=' the number of output units in the first layer of attention c')
'''
# default settings for Book-Crossing
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='book', help='which dataset to use')
parser.add_argument('--dim', type=int, default=4, help='dimension of entity and relation embeddings')
parser.add_argument('--n_hop', type=int, default=2, help='maximum hops')
parser.add_argument('--kge_weight', type=float, default=1e-2, help='weight of the KGE term')
parser.add_argument('--l2_weight', type=float, default=1e-5, help='weight of the l2 regularization term')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--n_epoch', type=int, default=10, help='the number of epochs')
parser.add_argument('--n_memory', type=int, default=32, help='size of ripple set for each hop')
parser.add_argument('--item_update_mode', type=str, default='plus_transform',
                    help='how to update item at the end of each hop')
parser.add_argument('--using_all_hops', type=bool, default=True,
                    help='whether using outputs of all hops or just the last hop when making prediction')
'''



args = parser.parse_args()


bayes_trials = Trials()




def model_thread(args,data_info):
    res=[]
    finish = threading.Event()
    def work_thread(event:threading.Event, *args):
        res.append(train(*args))
        event.set()
    t=threading.Thread(target=work_thread,args=(finish,args,data_info,True))
    t.start()
    finish.wait()
    return res[0]

def objective(hyperparameters):
    args.n_hop=int(hyperparameters['n_hop'])
    args.n_memory=int(hyperparameters['n_memory'])
    args.embed_size = int(hyperparameters['embed_size'])
    args.kge_weight = hyperparameters['kge_weight']
    args.l2_weight = hyperparameters['l2_weight']
    args.dim = int(hyperparameters['dim'])
    args.lr = hyperparameters['lr']
    data_info = load_data(args)
    acc = model_thread(args, data_info)

    return 1-acc

