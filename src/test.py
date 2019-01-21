from torch.utils.data import DataLoader
import logging
import torch
import numpy as np
from concurrent import futures
import time
import threading
from torch.autograd import Variable
from src.dataset import Expdata
from src.ripple_net import RippleNetPlus


logger = logging.getLogger()
logger.setLevel(level=logging.INFO)

batch_size = 1024
data_path = '../data'
dataset = 'movie'
n_hop = 3
n_memory = 32
dim = 16
kg_weight = 0.03
l2_weight = 0.03

dset = Expdata(data_path,dataset,n_hop, n_memory, dim)
# shape item:batch_size,dim, label h:[batch_size, n_hop, n_memory,dim],r:[batch_size, n_hop, n_memory,dim,dim],
# t:[batch_size, n_hop, n_memory,dim]
n_entity,n_relation = dset.get_n_enitity_relation()
model  = RippleNetPlus(n_hop, dim,n_entity,n_relation,kg_weight,l2_weight)
model.cuda()
early_stopping_cnt = 0
early_stopping_flag = False
best_acc = 0
optim = torch.optim.Adam(model.parameters())


dset.set_mode('train')
train_loader = DataLoader(
                    dset, batch_size=batch_size, shuffle=True
                )
model.train()
if not early_stopping_flag:
    total_acc = 0
    cnt = 0
    for batch_idx, data in enumerate(train_loader):
        optim.zero_grad()
        # noinspection PyRedeclaration
        vs,labels, hs, Rs, ts = data
        batch_size = vs.size()[0]
        vs = Variable(vs.cuda())
        labels = Variable(labels.to(torch.float32).cuda())
        hs = Variable(hs.cuda())
        Rs = Variable(Rs.cuda())
        ts = Variable(ts.cuda())

        loss = model.get_loss(vs,labels, hs, Rs, ts)
        loss.backward()

        cnt += batch_size
        optim.step()

        break

dset.set_mode('eval')
eval_loader = DataLoader(
    dset, batch_size=batch_size,shuffle=False
)
aucs = []
accs = []
for data in eval_loader:
    vs, labels, hs, Rs, ts = data
    batch_size = vs.size()[0]
    vs = Variable(vs.cuda())
    labels = Variable(labels.to(torch.float32).cuda())
    hs = Variable(hs.cuda())
    Rs = Variable(Rs.cuda())
    ts = Variable(ts.cuda())
    auc, acc = model.eval(vs,labels,hs,Rs,ts)
    aucs.append(auc)
    accs.append(acc)
acc = torch.mean(torch.stack(accs))
auc = torch.mean(torch.stack(aucs))
#logger.info('auc acc:{},{}'.format(auc,acc))

# dset = Expdata('../data','movie',2,10,4)
# # shape item:batch_size,dim, label h:[batch_size, n_hop, n_memory,dim],r:[batch_size, n_hop, n_memory,dim,dim],
# # t:[batch_size, n_hop, n_memory,dim]
# train_loader = DataLoader(
#                     dset, batch_size=100, shuffle=True
#                 )
#
# for data in train_loader:
#     break



# def worker(n,m):
#     for i in range(m,0,-1):
#         time.sleep(2)
#         print(n/i)
#
#     return 'haha'
# exector = futures.ThreadPoolExecutor(max_workers=1)
#
# f = exector.submit(worker,5,5)
# f.
# print(threading.enumerate())
# exector.shutdown()
# while True:
#     if f.done():
#         if f.exception():
#             print(f)
#             exector = futures.ThreadPoolExecutor(max_workers=1)
#             f = exector.submit(print,'haha')
#             continue
#         print('done')
#         break
