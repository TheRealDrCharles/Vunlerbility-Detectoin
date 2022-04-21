import torch
from torch import nn
import math

OFF_SLOPE=1e-3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# function to extract grad
def set_grad(var):
    def hook(grad):
        var.grad = grad
    return hook

class GradMod(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, input, other):
        """
        In the forward pass we receive a Tensor containing the input and return a
        Tensor containing the output. You can cache arbitrary Tensors for use in the
        backward pass using the save_for_backward method.
        """
        result = torch.fmod(input, other)
        ctx.save_for_backward(input, other)        
        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        x, y = ctx.saved_variables
        return grad_output * 1, grad_output * torch.neg(torch.floor_divide(x, y))
class PLSTMCell(nn.Module):
    def __init__(self, input_sz, hidden_sz):
        super().__init__()
        self.input_sz = input_sz   #200
        self.hidden_size = hidden_sz   #128
        self.Periods = nn.Parameter(torch.Tensor(hidden_sz, 1))
        self.Shifts = nn.Parameter(torch.Tensor(hidden_sz, 1))
        self.On_End = nn.Parameter(torch.Tensor(hidden_sz, 1))
        self.W = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_sz * 4))
        self.init_weights()
                
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        # Phased LSTM
        # -----------------------------------------------------
        nn.init.constant_(self.On_End, 0.05) # Set to be 5% "open"
        nn.init.uniform_(self.Shifts, 0, 100) # Have a wide spread of shifts
        # Uniformly distribute periods in log space between exp(1, 3)
        self.Periods.data.copy_(torch.exp((3 - 1) *
            torch.rand(self.Periods.shape) + 1))
        # -----------------------------------------------------
         
    def forward(self, t, x, ts,
                init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device), 
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states

        # PHASED LSTM
        # -----------------------------------------------------
        # Precalculate some useful vars
        shift_broadcast = self.Shifts.view(1, -1)
        period_broadcast = abs(self.Periods.view(1, -1))
        on_mid_broadcast = abs(self.On_End.view(1, -1)) * 0.5 * period_broadcast
        on_end_broadcast = abs(self.On_End.view(1, -1)) * period_broadcast                       
        
        def calc_time_gate(time_input_n):
            # Broadcast the time across all units
            t_broadcast = time_input_n.unsqueeze(-1)
            # Get the time within the period
            in_cycle_time = GradMod.apply(t_broadcast + shift_broadcast, period_broadcast)            

            # Find the phase
            is_up_phase = torch.le(in_cycle_time, on_mid_broadcast)
            is_down_phase = torch.gt(in_cycle_time, on_mid_broadcast)*torch.le(in_cycle_time, on_end_broadcast)


            # Set the mask
            sleep_wake_mask = torch.where(is_up_phase, in_cycle_time/on_mid_broadcast,
                                torch.where(is_down_phase,
                                    (on_end_broadcast-in_cycle_time)/on_mid_broadcast,
                                        OFF_SLOPE*(in_cycle_time/period_broadcast)))
            return sleep_wake_mask
        
        def calc_time_gate(time_input_n):
            # Broadcast the time across all units
            t_broadcast = time_input_n.unsqueeze(-1)
            # Get the time within the period
            in_cycle_time = GradMod.apply(t_broadcast + shift_broadcast, period_broadcast)            

            # Find the phase
            is_up_phase = torch.le(in_cycle_time, on_mid_broadcast)
            is_down_phase = torch.gt(in_cycle_time, on_mid_broadcast)*torch.le(in_cycle_time, on_end_broadcast)


            # Set the mask
            sleep_wake_mask = torch.where(is_up_phase, in_cycle_time/on_mid_broadcast,
                                torch.where(is_down_phase,
                                    (on_end_broadcast-in_cycle_time)/on_mid_broadcast,
                                        OFF_SLOPE*(in_cycle_time/period_broadcast)))
            return sleep_wake_mask
        # -----------------------------------------------------

        HS = self.hidden_size
        old_c_t, old_h_t = c_t, h_t
        x_t = x[:, t, :]
        t_t = ts[:, t]
        
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
        # PHASED LSTM
        # -----------------------------------------------------
        # Get time gate openness
        sleep_wake_mask = calc_time_gate(t_t)
        # Sleep if off, otherwise stay a bit on
        c_t = sleep_wake_mask*c_t + (1. - sleep_wake_mask)*old_c_t
        h_t = sleep_wake_mask*h_t + (1. - sleep_wake_mask)*old_h_t
        # -----------------------------------------------------
        return h_t, c_t

class BiPLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.plstm_cell_forward = PLSTMCell(self.input_sz, self.hidden_size).to(device)
        self.plstm_cell_backward = PLSTMCell(self.input_sz, self.hidden_size).to(device)
    def forward(self, x, ts,
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
            hs_forward, cs_forward = self.plstm_cell_forward(t, x, ts, (hs_forward, cs_forward))
            forward.append(hs_forward.unsqueeze(0))
        # Backward
        for t in reversed(range(seq_sz)):
            hs_backward, cs_backward = self.plstm_cell_backward(t, x, ts, (hs_backward, cs_backward))
            backward.append(hs_backward.unsqueeze(0))
        
        
        h_t, c_t = torch.cat((hs_forward, hs_backward), 1), torch.cat((cs_forward, cs_forward), 1)
        
        hidden_seq = []
        for fwd, bwd in zip(forward, backward):
            hidden_seq.append(torch.cat((fwd, bwd), 2))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)
    
class DLSTM_python_net(nn.Module):
    def __init__(self, weights):
        super(DLSTM_python_net, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=False)
        self.bplstm = BiPLSTM(200, 128).to(device)
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
        x, (hn, cn) = self.bplstm(x, ts)
        x = self.drop_out1(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.hidden1(x))
        x = x.view(x.size(0), -1)
        x = self.hidden2(x)
        x = x.view(x.size(0), -1)
        x = self.hidden3(x)
        return x
    
class DLSTM_python_Feature_Net(nn.Module):
    def __init__(self, net, weights):
        super(DLSTM_python_Feature_Net, self).__init__()
        self.embedding = net.embedding
        self.blstm = net.bplstm
        self.drop_out1 = net.drop_out1
        self.maxpool = net.maxpool
        self.hidden1 = net.hidden1
        self.relu = net.relu
    def forward(self, x):
        ts = x.to(device)
        x = self.embedding(x).float()
        x, (hn, cn) = self.blstm(x, ts)
        x = self.drop_out1(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.hidden1(x))
        return x