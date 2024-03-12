from .helpers import *
from .layers import LinearLoRAMergedLayer, LinearDoRAMergedLayer

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantDotProductGraphAttention(nn.module):
    '''
    Quantized location based attention utilizing adjacency matrix.
    Args:
    - dropout (float): probability for dropout layer
    '''
    def __init__(self, dropout=0.1):
        super(QuantDotProductGraphAttention).__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, adj_matrix, mask=None):
      '''
      Computes attention scores using location-based attention.
      Args:
      - q (torch.Tensor): Shape => (batch_size, num_nodes, in_features per node)
      - k (torch.Tensor): Shape => (batch_size, num_nodes, in_features per node)
      - v (torch.Tensor): Shape => (batch_size, num_nodes, in_features per node)
      - adj_matrix (numpy array) Shape => (batch_size, num_nodes, num_nodes)
      Returns:
      - output (torch.Tensor): Shape => (batch_size, num_nodes, out_features per node)
      - attn (torch.Tensor): attention weights 
      '''
      attn = quantized_matmul(q, k.transpose(2,3))
      if mask:
          attn = attn.masked_fill(adj_matrix == 0, float('-inf'))

      #TODO: improve   
      attn = dequantize(self.dropout(quantized_softmax(attn, dim=-1)))
      output = torch.matmul(attn, v)
      return output, attn

class DoRAGraphSelfAttention(nn.Module):
  def __init__(self, in_features, out_features, num_heads):
    super(DoRAGraphSelfAttention, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.num_heads = num_heads
    self.dora_query = LinearDoRAMergedLayer(in_features, out_features, bias=False)
    self.dora_key = LinearDoRAMergedLayer(in_features, out_features, bias=False)
    self.dora_value = LinearDoRAMergedLayer(in_features, out_features, bias=False)
    self.dropout = nn.Dropout(0.1)

  def forward(self, x, adj_matrix):
    '''
    Computes self-attention with graph and location based attention using DoRA Linear Layers
    Args:
    - x (torch.Tensor): Shape => (batch_size, num_nodes, in_features per node)
    - adj_matrix (numpy array) Shape => (batch_size, num_nodes, num_nodes)
    Returns:
    - output (torch.Tensor): Shape => (batch_size, num_nodes, out_features per node)
    - attn (torch.Tensor): attention weights
    '''
    bs, num_nodes, _ = x.size()
    query = self.dora_query(x).view(bs, num_nodes, self.num_heads,
                                      self.out_features // self.num_heads)
    key = self.dora_key(x).view(bs, num_nodes, self.num_heads,
                                      self.out_features // self.num_heads)
    value = self.dora_value(x).view(bs, num_nodes, self.num_heads,
                                      self.out_features // self.num_heads)

    output, attn = QuantDotProductGraphAttention(query, key, value, adj_matrix)
    return output, attn 
  
class LoRAGraphSelfAttention(nn.Module):
  def __init__(self, in_features, out_features, num_heads):
    super(LoRAGraphSelfAttention, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.num_heads = num_heads
    self.lora_query = LinearLoRAMergedLayer(in_features, out_features, bias=False)
    self.lora_key = LinearLoRAMergedLayer(in_features, out_features, bias=False)
    self.lora_value = LinearLoRAMergedLayer(in_features, out_features, bias=False)
    self.dropout = nn.Dropout(0.1)

  def forward(self, x, adj_matrix):
    '''
    Computes self-attention with graph and location based attention using LoRA Linear Layers
    Args:
    - x (torch.Tensor): Shape => (batch_size, num_nodes, in_features per node)
    - adj_matrix (numpy array) Shape => (batch_size, num_nodes, num_nodes)
    Returns:
    - output (torch.Tensor): Shape => (batch_size, num_nodes, out_features per node)
    - attn (torch.Tensor): attention weights
    '''
    bs, num_nodes, _ = x.size()
    query = self.lora_query(x).view(bs, num_nodes, self.num_heads,
                                        self.out_features // self.num_heads)
    key = self.lora_key(x).view(bs, num_nodes, self.num_heads,
                                        self.out_features // self.num_heads)
    value = self.lora_value(x).view(bs, num_nodes, self.num_heads,
                                        self.out_features // self.num_heads)
    output, attn = QuantDotProductGraphAttention(query, key, value, adj_matrix)
    return output, attn 

'''Test:

if __name__ == __main__:
    sentence_embeddings = torch.tensor([
    [[0.1, 0.2, 0.3, 0.4],
     [0.2, 0.3, 0.4, 0.5],
     [0.3, 0.4, 0.5, 0.6]],

    [[0.4, 0.5, 0.6, 0.7],
     [0.5, 0.6, 0.7, 0.8],
     [0.6, 0.7, 0.8, 0.9]]
])

adj_matrix = torch.ones(2, 3, 3)
n_features = 4
n_heads = 2

lora_graph_attn_module = LoRAGraphSelfAttention(n_features, n_features, n_heads)
lora_output = lora_graph_attn_module(sentence_embeddings, adj_matrix)

print("Output shape:", lora_output.shape)
print("Output tensor:")
print(lora_output)

dora_graph_attn_module = DoRAGraphSelfAttention(n_features, n_features, n_heads)
dora_output = dora_graph_attn_module(sentence_embeddings, adj_matrix)

print("Output shape:", dora_output.shape)
print("Output tensor:")
print(dora_output)
'''

