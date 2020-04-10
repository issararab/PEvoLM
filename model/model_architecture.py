import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch.nn as nn
import torch.nn.functional as F
import gettext as _

class Inducer(torch.nn.Module):
    def __init__(self,device_resource,input_size=512*2+20, hidden_size=256 ,num_layers=1 ,bi_directional=False
                 ,predict_next_pssm=True, predict_next_aa=False):
        super(Inducer, self).__init__()
        ############################################################################
        self.device = device_resource
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 1
        if bi_directional:
            self.num_directions = 2
        self.predict_next_pssm = predict_next_pssm
        self.predict_next_aa = predict_next_aa
        ##Recurrent layers
        self.lstm_block = nn.LSTM(input_size, hidden_size, num_layers=self.num_layers ,bidirectional= bi_directional
                                  ,batch_first=True)
        ##Linear layer
        #self.fc_layer = nn.Linear(hidden_siz e *self.num_directions, 256, bias=True)
        #nn.init.kaiming_normal_(self.fc_layer.weight, nonlinearity='relu')
        if self.predict_next_pssm:
            self.fc_NEXT_PSSM = nn.Linear(hidden_size, 20, bias=True)
        if self.predict_next_aa:
            self.fc_NEXT_AA = nn.Linear(hidden_size, 20, bias=True)
        # nn.init.xavier_uniform_(self.fc.weight)
        ############################################################################
    def forward(self, x ,seq_lens ,h_BPTT = 0 ,c_BPTT = 0 , processed_seqs=0 ,trc_BPTT = True):
        ############################################################################
        """
        The forward pass of the model architecture.

        Inputs:
        - x: a Tensor of size B x T x *, where B is the batch size and T the length of the longets sequence
        - seq_lens: a List of the original lengths of the sequences
        - h_BPTT: Hidden State Tensor passed from the forward pass of the last residue in the previous sequence chunk
                    in case of Truncated BPPTT. Default is zero if it is the beginning of the sequence
        - c_BPTT: Cell State Tensor passed from the forward pass of the last residue in the previous sequence chunk
                    in case of Truncated BPPTT. Default is zero if it is the beginning of the sequence
        - processed_seqs: The number of fully processed sequences during the last chunk of the mini-batch.
                    Used to slice the Hidden and Cell State Tensors.
        - trc_BPTT: Boolean to indicate that we are training using trancated BPTT
        """
        ############################################################################
        ## pack_padded_seq giving it the input_size lengths
        x_packed = pack_padded_sequence(x, seq_lens, batch_first=True, enforce_sorted=False)
        ## feed to LSTM layers
        ##Handle initialization of the trancated BPTT vs Full BP
        if trc_BPTT and type(h_BPTT) is not int:
            h0, c0 = h_BPTT.detach(), c_BPTT.detach()
        else:
            h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(self.device)
            c0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(self.device)
        ##Architecture core
        rnn_out, (hn, cn) = self.lstm_block(x_packed, (h0[:, : -processed_seqs or None], c0[:,: -processed_seqs or None]))  # Shape (batch_size, seq_length, hidden_size)
        # rnn_out = F.relu(self.fc_layer(rnn_out.data))
        rnn_out = rnn_out.data
        ## Multi-task branching
        if self.predict_next_pssm and self.predict_next_aa:
            ### feed fc + softmax for NEXT_PSSM path
            packed_out_NEXT_PSSM = self.fc_NEXT_PSSM(rnn_out)
            packed_out_NEXT_PSSM = F.softmax(packed_out_NEXT_PSSM, dim=1)
            ### feed fc + softmax for NEXT_AA path
            packed_out_NEXT_AA = self.fc_NEXT_AA(rnn_out)
            ##packed_out_NEXT_AA = F.softmax(packed_out_NEXT_AA, dim=1)
            return packed_out_NEXT_PSSM, packed_out_NEXT_AA, hn, cn
        elif self.predict_next_pssm:
            ### feed fc + softmax for NEXT_PSSM path
            packed_out_NEXT_PSSM = self.fc_NEXT_PSSM(rnn_out)
            packed_out_NEXT_PSSM = F.softmax(packed_out_NEXT_PSSM, dim=1)
            return packed_out_NEXT_PSSM, _, hn, cn
        else:
            ### feed fc + softmax for NEXT_AA path
            packed_out_NEXT_AA = self.fc_NEXT_AA(rnn_out)
            return _, packed_out_NEXT_AA, hn, cn
        ############################################################################

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Best model found. Saving model... %s' % path)
        torch.save(self.state_dict(), path)