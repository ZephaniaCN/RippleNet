
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from sklearn.metrics import roc_auc_score


logger = logging.getLogger()

class InputModule(nn.Module):
    def __init__(self,n_entity,n_relation,dim):
        super(InputModule, self).__init__()
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = dim
        self.entity_emb_matrix = torch.tensor(np.random.randn(self.n_entity, self.dim), dtype=torch.float32).cuda()
        self.relation_emb_matrix = torch.tensor(np.random.randn(self.n_relation, self.dim, self.dim), dtype=torch.float32).cuda()
        torch.nn.init.xavier_uniform_(self.entity_emb_matrix)
        torch.nn.init.xavier_uniform_(self.relation_emb_matrix)
        self.entity_emb_matrix.requires_grad = True
        self.relation_emb_matrix.requires_grad = True
    def forward(self, h_i,R_i,t_i,v_i):
        batch_size, n_hop,n_memory =h_i.size()
        h_i=h_i.view(-1)
        vs=self.entity_emb_matrix[v_i.long()]
        hs=self.entity_emb_matrix[h_i.long()].view(batch_size,n_hop,n_memory,-1)
        Rs=self.relation_emb_matrix[R_i.long()].view(batch_size,n_hop,n_memory,self.dim,-1)
        ts=self.entity_emb_matrix[t_i.long()].view(batch_size,n_hop,n_memory,-1)
        return hs,Rs,ts,vs
    # def get_loss(self,kg_weight,l2_weight):
    #     base_loss = 0
    #     l2_loss = 0
    #     for i in range(self.n_relation):
    #         ERE =  self.entity_emb_matrix.mm(self.relation_emb_matrix[i].mm(self.entity_emb_matrix.t()))
    #         base_loss += torch.dist(self.indicator[i],ERE) ** 2
    #         l2_loss += self.relation_emb_matrix[i].norm()
    #     loss = kg_weight * base_loss + l2_weight * (self.entity_emb_matrix.norm() * 2 + l2_loss)
    #     return loss





# class QuestionModule(nn.Module):
#     pass


class EpisodicMemory(nn.Module):
    def __init__(self,dim,n_hop,gru_use:bool=False):
        super(EpisodicMemory, self).__init__()
        self.hidden_size = dim
        self.dim = dim
        self.W1 = nn.Linear(4 * self.dim, self.hidden_size)
        self.W2 = nn.Linear(self.hidden_size, 1)

        init.xavier_normal_(self.W1.weight)
        init.xavier_normal_(self.W2.weight)

        self.gru_use = gru_use


        if not gru_use:
            self.W_mems = [nn.Linear(3 * dim, dim).cuda() for _ in range(n_hop)]
            for W_mem in self.W_mems:
                init.xavier_normal_(W_mem.weight)
        else:
            self.W_mems = [torch.nn.GRUCell(3 * dim, dim, bias=True).cuda() for _ in range(n_hop)]

    def gateMatrix(self, Rh, v, prev_mem):
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
        Z = self.W2(torch.tanh(self.W1(z)))
        #Z.size() = (batch_size,n_memory)
        Z = Z.view(-1,Rh.size()[1])
        # batch_size,n_memory
        g = torch.unsqueeze(torch.softmax(Z,dim=1),dim=2)

        return g

    def forward(self, Rh,v, t,prev_mem,hop_i):
        g = self.gateMatrix(Rh, v, prev_mem)
        logger.debug('g.size():{}'.format(g.size()))
        # [batch_size,1]
        o = self.attention_gate(t, g)
        logger.debug('o.size():{}'.format(o.size()))
        # update memory
        if(self.gru_use):
            o,m = self.W_mems[hop_i](o,prev_mem)
        else:
            concat = torch.cat([prev_mem.squeeze(1), o, v.squeeze(1)], dim=1)
            next_mem = F.relu(self.W_mems[hop_i](concat))
            #next_mem = next_mem.unsqueeze(1)
        return next_mem, o

    def attention_gate(self,t,g):
        #batch_size,n_memory,dim
        o = torch.sum(t * g, dim=1,keepdim=False)
        return o

class AnswerModule(nn.Module):
    def __init__(self, dim):
        super(AnswerModule, self).__init__()
        self.hidden_size = dim
        self.dim = dim
        self.z = nn.Linear(2 * self.hidden_size, self.dim)
        init.xavier_normal_(self.z.weight)
        self.dropout = nn.Dropout(0.1)

    def forward(self, M, v):
        M = self.dropout(M)
        concat = torch.cat([M, v], dim=1)
        #[batch_size,dim*2]
        u = self.z(concat)
        predict = torch.sum(v * u, dim=1)
        return predict


class RippleNetPlus(nn.Module):
    def __init__(self,n_hop,dim,n_entity,n_relation,kge_weight,l2_weight):
        super(RippleNetPlus, self).__init__()
        self.n_hop = n_hop
        self.dim = dim
        self.kge_weight=kge_weight
        self.l2_weight=l2_weight
        self.input_module= InputModule(n_entity,n_relation,self.dim)
        self.memory = EpisodicMemory(self.dim,self.n_hop,gru_use=False)
        self.answer_module = AnswerModule(self.dim)
    def forward(self, hs,Rs,ts,vs):
        '''
            hs.size() = [batch_size, n_hop, n_memory,dim],
            Rs.size():[batch_size, n_hop, n_memory,dim,dim]
            ts.size() = [batch_size, n_hop, n_memory,dim],
            v.size() = [batch_size, dim]
        '''
        M = vs
        o = vs
        batch_size,n_hop,n_memory,dim = hs.size()
        for hop in range(n_hop):
            # [batch_size, n_memory,dim,dim],
            R= Rs[:,hop]
            logger.debug('R.size():{}'.format(R.size()))
            # [batch_size, n_memory,dim],
            h= hs[:,hop]
            logger.debug('h.size():{}'.format(h.size()))
            # [batch_size, n_memory,dim,dim],
            t= ts[:,hop]
            logger.debug('t.size():{}'.format(t.size()))
            Rh_tmp = []
            for i in range(batch_size):
                for j in range(n_memory):
                    Rh_tmp.append(torch.squeeze(R[i, j].mm(h[i, j].unsqueeze(1))))
            Rh = torch.stack(Rh_tmp).view(batch_size, n_memory, -1)
            M, o = self.memory(Rh, o, t, M, hop)
        preds = self.answer_module(M, vs)
        return preds
    def get_loss(self,v_i,labels, h_i,R_i, t_i):
        hs, Rs, ts, vs = self.input_module(h_i, R_i, t_i, v_i)
        output = self.forward(hs, Rs,ts,vs)

        base_loss = self.cal_base_loss(output, labels)

        kg_loss,l2_loss = self.cal_kg_loss(hs,Rs,ts)

        logger.debug('kge_loss:{}'.format(kg_loss.item()))
        loss = base_loss + kg_loss +l2_loss

        return loss

    def cal_base_loss(self,output,labels):
        bce_loss_fn = torch.nn.BCEWithLogitsLoss(reduce=True)
        base_loss = bce_loss_fn(output, labels)
        logger.debug('base_loss:{}'.format(base_loss.item()))
        return base_loss


    def cal_kg_loss(self,hs,Rs,ts):
        batch_size, n_hop, n_memory, dim = hs.size()
        tmp = []
        for j in range(n_hop):
            for i in range(batch_size):
                for k in range(n_memory):
                    tRh = ts[i, j, k].unsqueeze(0).mm(Rs[i, j, k].mm(hs[i, j, k].unsqueeze(1)))
                    tmp.append(tRh)
        kg_loss = torch.sigmoid(torch.stack(tmp)).view(n_hop,-1).mean(1).sum()

        l2_loss = 0
        for hop in range(n_hop):
            l2_loss += torch.mean(torch.sum(Rs[:, hop] * Rs[:, hop], dim=[2, 3]))
            l2_loss += torch.mean(torch.sum(hs[:, hop] * hs[:, hop], dim=2))
            l2_loss += torch.mean(torch.sum(ts[:, hop] * ts[:, hop], dim=2))
        l2_loss = self.l2_weight * l2_loss

        return kg_loss,l2_loss

    def eval(self,v_i,labels, h_i,R_i, t_i):
        hs, Rs, ts, vs = self.input_module(h_i, R_i, t_i, v_i)
        output = self.forward(hs, Rs,ts,vs)
        predict = torch.floor(output+0.5)
        acc = torch.mean(torch.eq(predict,labels).to(torch.float32))
        auc = roc_auc_score(y_true=labels.cpu().detach().numpy(),y_score=output.cpu().detach().numpy())

        return auc, acc


