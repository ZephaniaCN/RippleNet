
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.utils.data import DataLoader

logger = logging.getLogger()

class InputModule(nn.Module):
    pass

class QuestionModule(nn.Module):
    pass

class EpisodicMemory(nn.Module):
    pass

class AnswerModule(nn.Module):
    pass
class RippleNetPlus(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_hop=3, qa=None):
        super(RippleNetPlus, self).__init__()
        self.num_hop = num_hop
        self.qa = qa
        self.word_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0, sparse=True).cuda()
        init.uniform(self.word_embedding.state_dict()['weight'], a=-(3 ** 0.5), b=3 ** 0.5)
        self.criterion = nn.CrossEntropyLoss(size_average=False)

    def forward(self, contexts, questions):
        pass
    def get_loss(self, contexts, questions, targets):
        pass
