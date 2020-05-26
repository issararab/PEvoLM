import math
import torch
from torch.nn import init
import torch.jit as jit
from torch.nn import Parameter
from torch import Tensor
from typing import List, Tuple
import torch.nn.functional as F
import numbers

class LSTMPCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, output_projection_size, dropout_rate = 0.0, memory_cell_clip_value=3.0, state_projection_clip_value=5.0, input_projection_size = 1024):
        super(LSTMPCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_projection_size = input_projection_size
        self.output_projection_size = output_projection_size
        self.dropout_rate = torch.jit.Attribute(dropout_rate,float)
        self.memory_cell_clip_value = torch.jit.Attribute(memory_cell_clip_value,float)
        #self.memory_cell_clip_value = Parameter(memory_cell_clip_value)
        self.state_projection_clip_value = torch.jit.Attribute(state_projection_clip_value,float)
        self.input_projection = torch.nn.Linear(input_size, input_projection_size, bias=True)
        self.input_projection_non_leaniarity = torch.nn.LeakyReLU(0.1)
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_projection_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, output_projection_size))
        self.weight_hr = Parameter(torch.randn(output_projection_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))
        self.init_weights()


    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        # input: batch_size * input_size
        # state: hx -> batch_size * projection_size
        #        cx -> batch_size * hidden_size
        # state cannot be None
        hx, cx = state
        input = self.input_projection_non_leaniarity(self.input_projection(input))
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        #memory
        cy = (forgetgate * cx) + (ingate * cellgate)
        ##Clip the memory cell before output
        if self.memory_cell_clip_value:
            cy = torch.clamp(cy,
                             -self.memory_cell_clip_value,
                             self.memory_cell_clip_value)
        hy = outgate * torch.tanh(cy)
        if self.dropout_rate:
            #Apply dropout in the projection layer
            F.dropout(hy, p=self.dropout_rate, training=self.training, inplace=True)
        hy = torch.mm(hy, self.weight_hr.t())
        #Clip the projected hidden state
        if self.state_projection_clip_value:
            hy = torch.clamp(hy,
                             -self.state_projection_clip_value,
                             self.state_projection_clip_value)

        return hy, (hy, cy)

    def init_weights(self):
        stdv_ih = math.sqrt(6.0 / (self.hidden_size+self.input_projection_size))
        stdv_hh = math.sqrt(6.0 / (self.hidden_size + self.output_projection_size))
        #torch.nn.init.kaiming_uniform_(self.input_projection.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.input_projection.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
        #init.xavier_uniform_(self.weight_ih)
        #init.xavier_normal_(self.weight_ih)
        # init.orthogonal_(self.weight_ih)
        #init.xavier_uniform_(self.weight_hh)
        init.uniform_(self.weight_ih, -stdv_ih, stdv_ih)
        init.uniform_(self.weight_hh, -stdv_hh, stdv_hh)
        init.xavier_uniform_(self.weight_hr)
        init.constant_(self.bias_ih, 1.0)
        init.constant_(self.bias_hh, 1.0)

class LSTMPLayer(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, projection_size, dropout_rate):
        super(LSTMPLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.projection_size = projection_size
        self.cell = LSTMPCell(input_size=input_size, hidden_size=hidden_size, output_projection_size=projection_size,dropout_rate=dropout_rate)

    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        # state cannot be None
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state