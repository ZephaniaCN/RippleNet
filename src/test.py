from torch.utils.data import DataLoader
import logging
import torch
import numpy as np
from concurrent import futures
import time
import threading
from torch.autograd import Variable
from pathlib import Path
from tensorboardX import SummaryWriter
from src.dataset import Expdata
from src.ripple_net import RippleNetPlus
from src.trainer import Trainer


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
save_name = 'test'
save_path = Path('./models') /save_name
max_loss = 1000

writer = SummaryWriter()


dset = Expdata(data_path,dataset,n_hop, n_memory, dim)
n_entity,n_relation = dset.get_n_enitity_relation()
model  = RippleNetPlus(n_hop, dim,n_entity,n_relation,kg_weight,l2_weight)
model.cuda()
optim = torch.optim.Adam(model.parameters())


trainer = Trainer(dset,model,optim,max_loss)

loss = trainer.train(batch_size)
auc,acc = trainer.eval(batch_size,'eval')

writer.add_scalar('loss', loss, 1)
writer.add_scalar('auc', auc, 1)
writer.add_scalar('auc', auc, 1)

torch.save(model.state_dict(), save_path)



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
