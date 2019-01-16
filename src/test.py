from torch.utils.data import DataLoader
import logging
from concurrent import futures
import time
import threading
from src.dataset import Expdata

logger = logging.getLogger()
logger.setLevel(level=logging.DEBUG)























# dset = Expdata('../data','movie',2,10)
# # shape item, label h:[batch_size, n_hop, n_memory],r:[batch_size, n_hop, n_memory],t:[batch_size, n_hop, n_memory]
# train_loader = DataLoader(
#                     dset, batch_size=100, shuffle=True
#                 )
# for data in train_loader:
#     print(data)
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
