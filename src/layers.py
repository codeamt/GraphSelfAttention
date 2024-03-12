import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALayer(nn.Module):
    '''
    '''
    def __init__(self, in_dimension, out_dimension, rank, alpha):
        super(LoRALayer).__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(in_dimension, rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_dimension))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x

class LinearLoRAMergedLayer(nn.Module):
    '''

    '''
    def __init__(self, in_dimension, out_dimension, rank, alpha):
        super(LinearLoRAMergedLayer).__init__()
        linear = nn.Linear(in_dimension, out_dimension)
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)
        self.alpha = alpha

    def forward(self, x):
        lora = self.lora.A @ self.lora.B
        merged_weights = self.linear.weight + self.lora.alpha * lora.T
        x = F.linear(x, merged_weights, self.linear.bias)
        return x 
    
class LinearDoRAMergedLayer(nn.Module):
    '''

    '''
    def __init__(self, in_dimension, out_dimension, rank, alpha):
        super(LinearLoRAMergedLayer).__init__()
        linear = nn.Linear(in_dimension, out_dimension)
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)
        self.m = nn.Parameter(self.linear.weight.norm(p=2, dim = 0, keepdim = True))

    def forward(self, x):
        lora = self.lora.A @ self.lora.B
        numerator = self.linear.weight + self.lora.alpha * lora.T
        denominator = numerator.norm (p=2, dim=0, keepdim=True)
        direction = numerator / denominator 
        merged_weights = self.m + direction
        x = F.linear(x, merged_weights, self.linear.bias)
        return x