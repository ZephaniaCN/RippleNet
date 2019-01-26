from torch.autograd import Variable
from torch.utils.data import DataLoader
from logging import getLogger
import torch
logger = getLogger()
class Trainer():
    def __init__(self,dset,model,optim,max_loss):
        self.dataset = dset
        self.model = model
        self.optim = optim
        self.max_loss = max_loss

    def train(self, batch_size):
        logger.info('start training...')
        self.dataset.set_mode('train')
        train_loader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True
        )
        self.model.train()
        for batch_idx, data in enumerate(train_loader):
            logger.info('batch_idx: {}'.format(batch_idx))
            self.optim.zero_grad()
            # noinspection PyRedeclaration
            vs, labels, hs, Rs, ts = data
            batch_size = vs.size()[0]
            vs = Variable(vs.cuda())
            labels = Variable(labels.to(torch.float32).cuda())
            hs = Variable(hs.cuda())
            Rs = Variable(Rs.cuda())
            ts = Variable(ts.cuda())

            loss = self.model.get_loss(vs, labels, hs, Rs, ts)
            loss.backward()
            if loss > self.max_loss:
                break
            self.optim.step()
            logger.info('batch_idx: {}\tloss:{}'.format(batch_idx,loss.item()))
        logger.info('finish train...')
        return loss

    def eval(self, batch_size,model):
        logger.info('start eval...')
        self.dataset.set_mode(model)
        self.model.eval()
        eval_loader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=False
        )
        aucs = []
        accs = []
        for batch_idx, data in enumerate(eval_loader):
            vs, labels, hs, Rs, ts = data
            batch_size = vs.size()[0]
            vs = Variable(vs.cuda())
            labels = Variable(labels.to(torch.float32).cuda())
            hs = Variable(hs.cuda())
            Rs = Variable(Rs.cuda())
            ts = Variable(ts.cuda())
            auc, acc = self.model.eval(vs, labels, hs, Rs, ts)
            aucs.append(auc)
            accs.append(acc)
            logger.info('batch_idx: {}\tauc:{}\tacc:{}'.format(batch_idx, auc,acc.item()))
        acc = torch.mean(torch.stack(accs))
        auc = torch.mean(torch.stack(aucs))
        logger.info('finish eval...')
        return auc,acc

# class Experiement():
#     def __init__(self, trainer):
#
#     def exec(self):
#         self.trainer =
