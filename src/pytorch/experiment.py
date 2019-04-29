from torch.autograd import Variable
from torch.utils.data import DataLoader
from logging import getLogger
from src.pytorch.model.ripple_net_plus import RippleNetPlus
from src.pytorch.dataset import Expdata
from src.pytorch.reporter import Reporters
from src.pytorch.args import args_convert
from sklearn.metrics import roc_auc_score
import numpy as np

import torch
import nni
logger = getLogger()


class Trainer():
    def __init__(self,dataset_args, model_args, model,lr , batch_size, n_epoch,reporter_mode, use_hyperopt,hyper_key,eval_train=True ,eval=True, test=True):
        model_dict = {
            'ripple_net_plus': RippleNetPlus
        }
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.n_epoch = n_epoch

        self.dataset = Expdata(**dataset_args)
        n_entity, n_relation = self.dataset.get_n_enitity_relation()

        self.model = model_dict[model](**model_args, n_entity=n_entity, n_relation=n_relation)
        self.model.to(self.device)

        self.optim = torch.optim.Adam(self.model.parameters(),lr=lr)
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.eval = eval
        self.test = test
        self.eval_train = eval_train
        self.reporter = Reporters(reporter_mode,use_hyperopt,hyper_key)

    def __set_config_mode(self, mode):
        model_switch = {
            'train': self.model.train,
            'test': self.model.eval,
            'eval': self.model.eval,
            'eval_train':self.model.eval
        }
        if(mode=='eval_train'):
            self.dataset.set_mode('train')
        else:
            self.dataset.set_mode(mode)
        data_loader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True
        )
        model_switch[mode]()
        return data_loader


    def __epoch_train(self):
        device = self.device
        train_loader = self.__set_config_mode('train')
        for batch_idx, data in enumerate(train_loader):
            logger.info('batch_idx: {}'.format(batch_idx))
            self.optim.zero_grad()
            v_i, labels, h_i, R_i, t_i = data
            v_i, labels, h_i, R_i, t_i = v_i.to(device), labels.to(device),h_i.to(device), R_i.to(device), t_i.to(device)

            loss = self.model.get_loss(v_i, labels, h_i, R_i, t_i)
            loss.backward()
            self.optim.step()
            logger.info('batch_idx: {}\tloss:{}'.format(batch_idx,loss.item()))
        return loss
    def train(self):
        logger.info('start training...')
        for epoch in range(self.n_epoch):
            loss = self.__epoch_train()
            res = {'loss': {'train':loss}}
            if(self.eval):
                eval_res = self.__epoch_eval('eval')
                for key in eval_res.keys():
                    res[key] = {**res.setdefault(key,{}), 'eval': eval_res[key]}
            if(self.test):
                test_res = self.__epoch_eval('test')
                for key in test_res.keys():
                    res[key] = {**res.setdefault(key,{}), 'test': test_res[key]}
            if(self.eval_train):
                train_res = self.__epoch_eval('eval_train')
                for key in train_res.keys():
                    res[key] = {**res.setdefault(key,{}), 'train': train_res[key]}

            self.reporter.intermediate_report(res, epoch)
        logger.info('finish train...')
        self.reporter.fin_report()
        return res

    def __epoch_eval(self, mode):
        logger.info('start eval...')
        aucs=[]
        accs=[]
        for i in range(5):
            auc,acc=self.__single_epoch_eval(mode)
            aucs.append(auc)
            accs.append(acc)
        auc=np.mean(aucs)
        acc=np.mean(accs)
        logger.info('finish eval...')
        logger.info('auc:{},acc:{}'.format(auc, acc))

        return {'auc':auc,'acc':acc}


    def __single_epoch_eval(self, mode):
        device = self.device
        eval_loader = self.__set_config_mode(mode)
        aucs = []
        accs = []
        for batch_idx, data in enumerate(eval_loader):
            v_i, labels, h_i, R_i, t_i = data
            v_i, labels, h_i, R_i, t_i = v_i.to(device), labels.to(device),h_i.to(device), R_i.to(device), t_i.to(device)

            output = self.model(h_i, R_i, t_i, v_i)
            predict = torch.floor(output + 0.5)

            acc = torch.mean(torch.eq(predict, labels.to(torch.float32)).to(torch.float32))
            auc = roc_auc_score(y_true=labels.cpu().detach().numpy(), y_score=output.cpu().detach().numpy())
            aucs.append(auc)
            accs.append(acc)
        acc = torch.mean(torch.stack(accs))
        auc = np.mean(aucs)

        return auc, acc.item()


def run_exp(default_args):
    args = nni.get_next_parameter()
    args = {**default_args, **args}
    trainer=Trainer(**args_convert(args))
    trainer.train()