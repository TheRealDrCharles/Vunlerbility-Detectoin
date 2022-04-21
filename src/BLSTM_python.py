import torch
from torch import nn
import math

# Thanks to https://gist.github.com/piEsposito/c8d0b2e1cc8e3cd98266da6fd69d3147
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.W = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_sz * 4))
        self.init_weights()
                
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
         
    def forward(self, t, x, 
                init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device), 
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states
         
        HS = self.hidden_size
        x_t = x[:, t, :]
        # batch the computations into a single matrix multiplication
        gates = x_t @ self.W + h_t @ self.U + self.bias
        i_t, f_t, g_t, o_t = (
            torch.sigmoid(gates[:, :HS]), # input
            torch.sigmoid(gates[:, HS:HS*2]), # forget
            torch.tanh(gates[:, HS*2:HS*3]),
            torch.sigmoid(gates[:, HS*3:]), # output
        )
        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t
    
    
class BiLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.blstm_cell_forward = CustomLSTM(self.input_sz, self.hidden_size).to(device)
        self.blstm_cell_backward = CustomLSTM(self.input_sz, self.hidden_size).to(device)
    def forward(self, x,
               init_states=None):
        bs, seq_sz, _ = x.size()
        hs_forward = torch.zeros(x.size(0), self.hidden_size).to(device)
        cs_forward = torch.zeros(x.size(0), self.hidden_size).to(device)
        hs_backward = torch.zeros(x.size(0), self.hidden_size).to(device)
        cs_backward = torch.zeros(x.size(0), self.hidden_size).to(device)
        
        forward = []
        backward = []
        
        # Forward
        for t in range(seq_sz):
            hs_forward, cs_forward = self.blstm_cell_forward(t, x, (hs_forward, cs_forward))
            forward.append(hs_forward.unsqueeze(0))
        # Backward
        for t in reversed(range(seq_sz)):
            hs_backward, cs_backward = self.blstm_cell_backward(t, x, (hs_backward, cs_backward))
            backward.append(hs_backward.unsqueeze(0))
        
        
        h_t, c_t = torch.cat((hs_forward, hs_backward), 1), torch.cat((cs_forward, cs_forward), 1)
        
        hidden_seq = []
        for fwd, bwd in zip(forward, backward):
            hidden_seq.append(torch.cat((fwd, bwd), 2))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)
    
class BLSTM_python_net(nn.Module):
    def __init__(self, weights):
        super(BLSTM_python_net, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=False)
        self.blstm = BiLSTM(200 + 1, 128).to(device)
        self.drop_out1 = nn.Dropout()
        self.maxpool = nn.MaxPool1d(3, stride=2)
        self.hidden1 = nn.Linear(127*200, 128)
        self.hidden2 = nn.Linear(128,32)
        self.hidden3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        ts = x.to(device)
        x = self.embedding(x).float()
        combined_input = torch.cat((x, torch.unsqueeze(ts, dim=-1)), -1)
        x, (hn, cn) = self.blstm(combined_input)
        x = self.drop_out1(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.hidden1(x))
        x = x.view(x.size(0), -1)
        x = self.hidden2(x)
        x = x.view(x.size(0), -1)
        x = self.hidden3(x)
        return x
    
class BLSTM_python_Feature_Net(nn.Module):
    def __init__(self, net, weights):
        super(BLSTM_python_Feature_Net, self).__init__()
        self.embedding = net.embedding
        self.blstm = net.blstm
        self.drop_out1 = net.drop_out1
        self.maxpool = net.maxpool
        self.hidden1 = net.hidden1
        self.relu = net.relu
    def forward(self, x):
        ts = x.to(device)
        x = self.embedding(x).float()
        combined_input = torch.cat((x, torch.unsqueeze(ts, dim=-1)), -1)
        x, (hn, cn) = self.blstm(combined_input)
        x = self.drop_out1(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.hidden1(x))
        return x