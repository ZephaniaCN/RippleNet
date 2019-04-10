import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import logging
import numpy as np
from sklearn.metrics import roc_auc_score

logger = logging.getLogger()

class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

class InputModule(nn.Module):
    def __init__(self,n_entity,n_relation,dim):
        super(InputModule, self).__init__()
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = dim
        self.entity_emb_matrix = nn.Parameter(torch.tensor(np.random.randn(self.n_entity, self.dim), dtype=torch.float32), requires_grad=True)
        self.relation_emb_matrix = nn.Parameter(torch.tensor(np.random.randn(self.n_relation, self.dim, self.dim), dtype=torch.float32), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.entity_emb_matrix)
        torch.nn.init.xavier_uniform_(self.relation_emb_matrix)

    def forward(self, h_i,R_i,t_i,v_i):
        batch_size, n_hop,n_memory =h_i.size()
        h_i=h_i.view(-1)
        vs=self.entity_emb_matrix[v_i.long()]
        hs=self.entity_emb_matrix[h_i.long()].view(-1,n_hop,n_memory,self.dim)
        Rs=self.relation_emb_matrix[R_i.long()].view(-1,n_hop,n_memory,self.dim,self.dim)
        ts=self.entity_emb_matrix[t_i.long()].view(-1,n_hop,n_memory,self.dim)
        return hs,Rs,ts,vs


class AggregateFnc(nn.Module):
    def __init__(self, dim):
        super(AggregateFnc, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(4 * dim, dim), nn.Tanh())
        self.layer2 = nn.Linear(dim, 1)

        def weights_init(m):
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)

        self.layer1.apply(weights_init)
        init.xavier_normal_(self.layer2.weight)

    def forward(self, z):
        z = self.layer1(z)
        Z = self.layer2(z)
        return Z

class PropagateLayer(nn.Module):
    def __init__(self, dim, gru_use, aggregate,dropout):
        super(PropagateLayer, self).__init__()
        self.hidden_size = dim
        self.dim = dim
        self.aggregate = aggregate
        self.gru_use = gru_use

        if not gru_use:
            self.W_update =nn.Sequential(nn.Linear(3 * dim, dim), nn.Tanh)
            def weights_init(m):
                if isinstance(m, nn.Linear):
                    init.xavier_normal_(m.weight)
            self.W_update.apply(weights_init)
        else:
            self.dropout = torch.nn.Dropout(dropout)
            self.W_update = torch.nn.GRUCell(dim, dim, bias=True)


    def forward(self, Rh, v, prev_mem, t):
        # Rh.size() = (batch_size, n_memory, embedding_length=hidden_size)
        # v.size() = (batch_size, 1, embedding_length)
        # prev_mem.size() = (batch_size, 1, embedding_length)
        # z.size() = (batch_size, n_memory, 4*embedding_length)
        # g.size() = (batch_size, n_memory, 1)

        v = v.unsqueeze(1)
        prev_mem = prev_mem.unsqueeze(1)

        z = torch.cat([Rh * v, Rh * prev_mem, torch.abs(Rh - v), torch.abs(Rh - prev_mem)],
                      dim=2)
        # z.size() = (batch_size*n_memory, 4*embedding_length)
        z = z.view(-1, 4 * self.hidden_size)
        Z = self.aggregate(z)
        # Z.size() = (batch_size,n_memory)
        Z = Z.view(-1, Rh.size()[1])

        # attention
        # batch_size,n_memory
        g = torch.unsqueeze(torch.softmax(Z, dim=1), dim=2)
        # batch_size dim
        o = torch.sum(t * g, dim=1, keepdim=False)

        if self.gru_use:
            o = self.dropout(o)
            next_mem = self.W_update(o, prev_mem.squeeze(1))
        else:
            concat = torch.cat([prev_mem.squeeze(1), o, v.squeeze(1)], dim=1)
            next_mem = self.W_update(concat)
        return next_mem

class EpisodicMemory(nn.Module):
    def __init__(self,dim,n_hop, dropout, gru_use:bool=True):
        super(EpisodicMemory, self).__init__()
        self.dim = dim
        self.n_hop = n_hop
        self.aggregate = AggregateFnc(dim)
        self.propagates = [PropagateLayer(dim, gru_use, self.aggregate, dropout) for _ in range(n_hop)]
        self.propagates = ListModule(*self.propagates)

    def forward(self, hs, Rs, ts, vs):
        '''
            hs.size() = [batch_size, n_hop, n_memory,dim],
            Rs.size():[batch_size, n_hop, n_memory,dim,dim]
            ts.size() = [batch_size, n_hop, n_memory,dim],
            vs.size() = [batch_size, dim]
        '''
        M = vs
        batch_size, n_hop, n_memory, dim = hs.size()
        for hop in range(n_hop):
            # [batch_size, n_memory,dim,dim],
            R = Rs[:, hop]
            logger.debug('R.size():{}'.format(R.size()))
            # [batch_size, n_memory,dim,1],
            h = hs[:, hop].unsqueeze(3)
            logger.debug('h.size():{}'.format(h.size()))
            # [batch_size, n_memory,dim],
            t = ts[:, hop]
            logger.debug('t.size():{}'.format(t.size()))

            # [batch_size, n_memory, dim]
            Rh = torch.matmul(R, h).squeeze(3)


            M = self.propagates[hop](Rh, vs, M, t)


        return M

class AnswerModule(nn.Module):
    def __init__(self, dim):
        super(AnswerModule, self).__init__()
        self.hidden_size = dim
        self.dim = dim
        self.w = nn.Sequential(nn.Linear(self.dim, self.dim), nn.Sigmoid(), nn.Dropout(0.1))

    def forward(self, M,  v):
        M = self.w(M)
        predict = torch.sum(v * M, dim=1)
        return predict

class RippleNetPlus(nn.Module):

    def __init__(self, n_hop, dim, n_entity, n_relation, kge_weight, l2_weight,dropout):
        super(RippleNetPlus, self).__init__()
        self.n_hop = n_hop
        self.dim = dim
        self.kge_weight = kge_weight
        self.l2_weight = l2_weight
        self.input_module = InputModule(n_entity, n_relation, self.dim)
        self.memory = EpisodicMemory(self.dim, self.n_hop ,dropout=dropout,gru_use=True)
        self.answer_module = AnswerModule(self.dim)

    def forward(self, h_i, R_i, t_i, v_i):
        hs, Rs, ts, vs = self.input_module(h_i, R_i, t_i, v_i)
        M = self.memory(hs, Rs, ts, vs)
        preds = self.answer_module(M, vs)
        return preds

    def get_loss(self, v_i, labels, h_i, R_i, t_i):

        hs, Rs, ts, vs = self.input_module(h_i, R_i, t_i, v_i)
        M = self.memory(hs, Rs, ts, vs)
        output = self.answer_module(M, vs)

        base_loss = self.cal_base_loss(output, labels)

        kg_loss, l2_loss = self.cal_kg_loss(hs, Rs, ts)

        loss = base_loss + kg_loss + l2_loss

        return loss

    def cal_base_loss(self, output, labels):
        labels=labels.to(torch.float32)
        bce_loss_fn = torch.nn.BCEWithLogitsLoss(reduce=True)
        base_loss = bce_loss_fn(output, labels)
        logger.debug('base_loss:{}'.format(base_loss.item()))
        return base_loss

    def cal_kg_loss(self, hs, Rs, ts):

        kge_loss = 0
        for hop in range(self.n_hop):
            # [batch_size, n_memory, 1, dim]
            h_expanded = torch.unsqueeze(hs[:,hop], dim=2)
            # [batch_size, n_memory, dim, 1]
            t_expanded = torch.unsqueeze(ts[:,hop], dim=3)
            # Rs[ batch_size, n_memory, dim, dim]
            # hR[ batch_size, n_memory, 1, dim]
            # hRt[batch_size n_memory]
            hRt = torch.squeeze(torch.matmul(torch.matmul(h_expanded, Rs[:,hop]), t_expanded))
            kge_loss += torch.mean(torch.sigmoid(hRt))
        kge_loss = -self.kge_weight * kge_loss

        l2_loss = 0
        for hop in range(self.n_hop):
            l2_loss += torch.mean(torch.sum(Rs[:, hop] * Rs[:, hop], dim=[2, 3]))
            l2_loss += torch.mean(torch.sum(hs[:, hop] * hs[:, hop], dim=2))
            l2_loss += torch.mean(torch.sum(ts[:, hop] * ts[:, hop], dim=2))
        l2_loss = self.l2_weight * l2_loss

        return kge_loss, l2_loss
