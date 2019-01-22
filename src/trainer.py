from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch
class Trainer():
    def __init__(self,dset,model,optim,max_loss):
        self.dataset = dset
        self.model = model
        self.optim = optim
        self.max_loss = max_loss

    def train(self, batch_size):
        self.dataset.set_mode('train')
        train_loader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True
        )
        self.model.train()
        for batch_idx, data in enumerate(train_loader):
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
        return loss

    def eval(self, batch_size,model):
        self.dataset.set_mode(model)
        self.model.eval()
        eval_loader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=False
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
            auc, acc = self.model.eval(vs, labels, hs, Rs, ts)
            aucs.append(auc)
            accs.append(acc)
        acc = torch.mean(torch.stack(accs))
        auc = torch.mean(torch.stack(aucs))
        return auc,acc