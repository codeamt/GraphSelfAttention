from .attn import GraphSelfAttention
from .layers import LinearLoRAMergedLayer, LinearDoRAMergedLayer

import torch.nn as nn
import torch.nn.functional as F


class FFLoRATransformerLayer(nn.Module):
  '''
  
  '''
  def __init__(self, in_features, out_features, num_heads, rank=2, alpha=4):
    super(FFLoRATransformerLayer, self).__init__()
    self.attention = GraphSelfAttention(in_features, out_features, num_heads)
    self.lora1 = LinearLoRAMergedLayer(out_features, 4 * out_features, rank, alpha)
    self.lora2 = LinearLoRAMergedLayer(4 * out_features, out_features, rank, alpha)
    self.norm1 = nn.LayerNorm(out_features)
    self.norm2 = nn.LayerNorm(out_features)
    self.dropout = nn.Dropout(0.1)

  def forward(self, x, adj_matrix):
    '''
    '''
    attn_output = self.attention(x, adj_matrix)
    x = self.norm1(x + self.dropout(attn_output))
    linear_output = self.lora2(F.relu(self.lora1(x)))
    output = self.norm2(x + self.dropout(linear_output))
    return output
  
class FFDoRATransformerLayer(nn.Module):
  '''
  
  '''
  def __init__(self, in_features, out_features, num_heads, rank=4, alpha=8):
    super(FFDoRATransformerLayer, self).__init__()
    self.attention = GraphSelfAttention(in_features, out_features, num_heads)
    self.lora1 = LinearDoRAMergedLayer(out_features, 4 * out_features, rank, alpha)
    self.lora2 = LinearDoRAMergedLayer(4 * out_features, out_features, rank, alpha)
    self.norm1 = nn.LayerNorm(out_features)
    self.norm2 = nn.LayerNorm(out_features)
    self.dropout = nn.Dropout(0.1)

  def forward(self, x, adj_matrix):
    '''
    '''
    attn_output = self.attention(x, adj_matrix)
    x = self.norm1(x + self.dropout(attn_output))
    linear_output = self.lora2(F.relu(self.lora1(x)))
    output = self.norm2(x + self.dropout(linear_output))
    return output


class GraphAttnTransformer(nn.Module):
  '''

  '''
  def __init__(self, in_features, out_features, num_heads, num_layers, linear_type='dora'):
    super(GraphAttnTransformer, self).__init__()

    if linear_type == 'dora':
      self.layers = nn.ModuleList([FFDoRATransformerLayer(in_features, out_features, num_heads) for
                                 _ in range(num_layers)])
      
    self.layers = nn.ModuleList([FFLoRATransformerLayer(in_features, out_features, num_heads) for
                                 _ in range(num_layers)])

  def forward(self, x, adj_matrix):
    for layer in self.layers:
      x = layer(x, adj_matrix)
    return x


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

adj_matrices = torch.tensor([
    [[1, 1, 1],
     [1, 1, 1],
     [1, 1, 1]],

    [[1, 0, 1],
     [0, 1, 0],
     [1, 0, 1]]
])
n_features = 4
out_features = 4
n_heads = 2
n_layers = 2

graph_attn_transformer = GraphAttnTransformer(n_features, out_features, n_heads, n_layers)
output = graph_attn_transformer(sentence_embeddings, adj_matrices)

print("Output shape:", output.shape)
print("Output tensor:")
print(output)
'''