import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImprovedMarginRankingLoss(nn.Module):
    def __init__(self,margin):
        super(ImprovedMarginRankingLoss,self).__init__()
        self.margin = margin
        self.loss_func = torch.nn.MarginRankingLoss(margin)
        
    def forward(self,similarities):
        num_lines = similarities.shape[0]
        total_loss = 0
        avg_similarity = torch.mean(similarities)
        
        for i in range(num_lines):
            cap_similarity = similarities[i]
            target_value =1  if cap_similarity > avg_similarity else -1
            target = torch.tensor([target_value],dtype=torch.float).to(cap_similarity.device)
            loss = self.loss_func(cap_similarity.view(-1),avg_similarity.view(-1),target)
            total_loss += loss
            
        return total_loss / num_lines

class sparsity_loss(nn.Module):
    def __init__(self):
        super(sparsity_loss,self).__init__()
        
    def forward(self,pred,target):
        loss = torch.abs(torch.mean(pred) - target)
        return loss