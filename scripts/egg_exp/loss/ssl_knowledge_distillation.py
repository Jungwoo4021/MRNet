import torch
import torch.nn as nn

from .interface import Criterion

class LastHiddenMSE(Criterion):
    '''KD loss simply comparing the last outputs of teachers and students using MSE. 
    '''
    def __init__(self):
        super(Criterion, self).__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, x, label):
        x = x[:, -1, :, :]
        if label.size(1) != 1:
            label = label[:, -1, :, :]
        
        loss = self.mse(x, label)
        return loss

class DistilHubertKDLoss(Criterion):
    '''SSL KD loss function used in papaer 'Distilhubert: Speech representation learning by layer-wise distillation of hidden-unit bert'.
    This compute (l1_loss - cos_sim) in the hidden layers. 
    '''
    def __init__(self):
        super(DistilHubertKDLoss, self).__init__()

        self.log_sigmoid = nn.LogSigmoid()
        self.cos_sim = nn.CosineSimilarity(dim=1)

    def forward(self, x, label):
        bs, l, s, h = x.size()  # bs, num_layer, sequence_length, hidden_size

        # calculate L1-loss
        if label.size(1) != 3:
            label = label[:, [4, 8, 12], :, :]
        
        l1_loss = torch.abs(x.reshape(bs, l, s * h) - label.reshape(bs, l, s * h))
        l1_loss = torch.mean(l1_loss, dim=-1)

        # calculate cosine score
        cos_sim_loss = self.cos_sim(x.reshape(bs * l * s, h), label.reshape(bs * l * s, h))
        cos_sim_loss = self.log_sigmoid(cos_sim_loss).view(bs, l, s)
        cos_sim_loss = cos_sim_loss.sum(dim=2)

        loss = l1_loss - cos_sim_loss
        loss = loss.sum(dim=1)
        loss = loss.mean()

        return loss
    
class FitHubertKDLoss(Criterion):
    '''SSL KD loss function used in papaer 'Fithubert: Speech representation learning by layer-wise distillation of hidden-unit bert'.
    This compute (l1_loss - cos_sim) in the hidden layers. 
    '''
    def __init__(self, hintloss_weight):
        super(FitHubertKDLoss, self).__init__()

        self.hintloss_weight = hintloss_weight

    def forward(self, x, label):
        # drop conv output
        label = label[:, 1:, :, :]
        bs, l, s, h = label.size()  # bs, num_layer, sequence_length(149), hidden_size
        
        x_h = {}
        label_h = {}
        tot_loss = []

        # transform student output
        for i in range(l):
            x_h['%d'%i] = x[:, i, :, :].reshape(bs * s, h)
            label_h['%d'%i] = label[:, i, :, :].reshape(bs * s, h)

            # calculate l2-loss
            l2_loss = torch.mean((x_h['%d'%i] - label_h['%d'%i]) ** 2, dim=-1).reshape(bs, s)
            l2_loss = torch.mean(l2_loss, dim=-1)

            # multiply the hint loss function by the hint loss weight
            if i != l-1:
                l2_loss = l2_loss * self.hintloss_weight

            tot_loss.append(l2_loss)

        tot_loss = sum(tot_loss)
        tot_loss = torch.mean(tot_loss, dim=0)

        return tot_loss