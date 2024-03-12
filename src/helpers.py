import torch
import torch.nn.functional as F

def quantized_softmax(scores, dim, bits=8, range_scale=1.0):
  """
  Applies softmax using quantized arithmetic. 
  Args:
  scores (torch.Tensor)
  dim (int): dimention to apply softmax operation
  bits (int): specifies number of bits for quant precision
  range scale (float): scaling for quant range
  Returns:
  - quantized_result (torch.Tensor): quantized scores (for attention)
  """
  max_value = scores.max()
  min_value = scores.min()
  qmin = 0.
  qmax = 2.**bits - 1.
  scale = (range_scale * (qmax - qmin)) / (max_value - min_value)
  zero_point = qmax - max_value * scale
  quantized_scores = torch.round(scale * scores + zero_point)
  softmax_scores = F.softmax(quantized_scores / (2.**bits), dim=dim)
  return softmax_scores


def quantized_matmul(a, b, bits=8, range_scale=1.0):
  """
  Performs quantized matrix multiplication. 
  Args:
  a (torch.Tensor)
  b (torch.Tensor)
  bits (int): specifies number of bits for quant precision
  range scale (float): scaling for quant range
  Returns:
  - quantized_result (torch.Tensor): quantized matrix multiplication result
  """
  max_val_a = a.max()
  min_val_a = a.min()
  max_val_b = b.max()
  min_val_b = b.min()

  max_val = max(max_val_a, max_val_b)
  min_val = min(min_val_a, min_val_b)
  qmin = 0.
  qmax = 2.**bits - 1.

  scale = (range_scale * (qmax - qmin)) / (max_val - min_val)
  zero_point = qmax - max_val * scale
  quantized_a = torch.round(scale * a + zero_point)
  quantized_b = torch.round(scale * b + zero_point)

  quantized_result = torch.matmul(quantized_a, quantized_b) / (2.** bits)
  return quantized_result


def dequantize(result, bits=8, range_scale=1.0):
  """
  Returns quantized result to full precision space. 
  result (torch.Tensor)
  bits (int): quant bit precision
  range scale (float): scaling for quant range
  Returns:
  - dequantized_result (torch.Tensor): result in full precision
  """
  qmin = 0
  qmax = 2**bits - 1
  scale = (range_scale * (qmax- qmin)) / (2**bits)
  zero_point = qmax - qmin - scale
  dequantized_result = scale * result - zero_point
  return dequantized_result