import math
import torch 
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product(q, k, v, mask=None):
    dk = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dk)
    if mask is not None:
        scaled += mask
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, d_model, num_heads):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(input_dim, 3 * d_model)  
        self.linear_layer = nn.Linear(d_model, d_model)  

    def forward(self, x, mask=None):
        batch_size, sequence_length, _ = x.size()
        qkv = self.qkv_layer(x)   
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3, self.head_dim)          
        qkv = qkv.permute(0, 2, 1, 3, 4) 
        q, k, v = qkv.chunk(3, dim=-2)  
        values, attention = scaled_dot_product(q.squeeze(-2), k.squeeze(-2), v.squeeze(-2), mask)  
        values = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim) 
        out = self.linear_layer(values) 
        return out
    
class LayerNormalization(nn.Module):
    def __init__(self, parameter_shape, epsilon=1e-5):
        super().__init__()
        self.parameter_shape = parameter_shape
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(parameter_shape), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(parameter_shape), requires_grad=True)
        
    def forward(self, inputs):
        mean = inputs.mean(dim=-1, keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (inputs - mean) / std
        return self.gamma * y + self.beta
 
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)
    
    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, nums_heads, drop_prob):
        super().__init__()
        self.attention = MultiheadAttention(d_model, d_model, nums_heads)
        self.norm1 = LayerNormalization(parameter_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm2 = LayerNormalization(parameter_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)
        
    def forward(self, x, mask=None):
        residual_x = x
        x = self.attention(x, mask)
        x = self.dropout1(x)
        x = self.norm1(x + residual_x)
        
        residual_x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual_x)
        return x        
    
class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, nums_heads, drop_prob, nums_layers):
        super().__init__()        
        self.layers = nn.Sequential(*[EncoderLayer(d_model, ffn_hidden, nums_heads, drop_prob)
                                      for _ in range(nums_layers)])
    
    def forward(self, x, mask=None):
        return self.layers(x)    

d_model = 512
num_heads = 8
drop_prob = 0.1     
batch_size = 30
max_sequence_length = 200
ffn_hidden = 2048
num_layers = 5

encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)    
x = torch.randn((batch_size, max_sequence_length, d_model))
out = encoder(x)
print("Output Shape:", out.shape)  
