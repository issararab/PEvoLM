import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch.nn.functional as F
import gettext as _
from model.lstm_with_projection import LSTMPLayer


class Inducer(torch.nn.Module):
    def __init__(self,device_resource, hidden_size=4096,projection_size=512,dropout_rate= 0.0,predict_next_pssm=True, predict_next_aa=False):
        super(Inducer, self).__init__()
        self.device = device_resource
        self.hidden_size = hidden_size
        self.projection_size = projection_size
        ##Multi-task config.
        self.predict_next_pssm = predict_next_pssm
        self.predict_next_aa = predict_next_aa
        ##Recurrent layers
        self.LSTMP_L1 = LSTMPLayer(512+20, self.hidden_size, self.projection_size, dropout_rate=dropout_rate)
        #self.LSTMP_L2 = LSTMPLayer(512*3+20+256, self.hidden_size, self.projection_size, dropout_rate=dropout_rate)
        self.LSTMP_L2 = LSTMPLayer(512 * 2 + 256, self.hidden_size, self.projection_size,
                                   dropout_rate=dropout_rate)
        if self.predict_next_pssm:
            self.fc_NEXT_PSSM = nn.Linear(self.projection_size, 20, bias=True)
        if self.predict_next_aa:
            self.fc_NEXT_AA = nn.Linear(self.projection_size, 20, bias=True)


    def forward(self, uncon_x, con_x, seq_lens, lstmp1_tbptt_state = None ,lstmp2_tbptt_state = None, processed_seqs=0):
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
        print('\tInput size: {}'.format(input.shape))
        uncon_x, con_x = uncon_x.to(self.device), con_x.to(self.device)
        ## feed to LSTM layers
        ##Handle initialization of the trancated BPTT
        if lstmp1_tbptt_state is None:
            h1 = torch.zeros(uncon_x.size(1), self.projection_size, requires_grad=False).to(self.device)
            c1 = torch.zeros(uncon_x.size(1), self.hidden_size, requires_grad=False).to(self.device)
            h2 = torch.zeros(con_x.size(1), self.projection_size, requires_grad=False).to(self.device)
            c2 = torch.zeros(con_x.size(1), self.hidden_size, requires_grad=False).to(self.device)
        else:
            h1, c1, h2, c2 = lstmp1_tbptt_state[0].detach(), lstmp1_tbptt_state[1].detach(), lstmp2_tbptt_state[0].detach(), lstmp2_tbptt_state[1].detach()

        ##Architecture core
        ## pack_padded_seq giving it the input_size lengths
        #uncon_x_packed = pack_padded_sequence(uncon_x, seq_lens, batch_first=True, enforce_sorted=True)
        lstmp1_output, lstmp1_state = self.LSTMP_L1(uncon_x,[h1[: -processed_seqs or None,:].contiguous(), c1[: -processed_seqs or None,:].contiguous()])  # Shape (batch_size, seq_length, hidden_size)

        ##Concatinate with the contextualized SeqVec embedding
        #output = torch.cat((lstmp1_output,uncon_x, con_x), dim=2)
        output = torch.cat((lstmp1_output, con_x), dim=2)
        ##Run the second layer forward LSTM with projection
        output, lstmp2_state = self.LSTMP_L2(output, [h2[: -processed_seqs or None,:].contiguous(), c2[: -processed_seqs or None,:].contiguous()])  # Shape (batch_size, seq_length, hidden_size)
        ## Skip connection, just adding the input to the output.
        output += lstmp1_output
        ## pack_padded_seq giving it the input_size lengths
        output = pack_padded_sequence(output, seq_lens, enforce_sorted=True)
        ## Multi-task branching
        if self.predict_next_pssm and self.predict_next_aa:
            ### feed fc + softmax for NEXT_PSSM path
            packed_out_NEXT_PSSM = self.fc_NEXT_PSSM(output.data)
            packed_out_NEXT_PSSM = F.softmax(packed_out_NEXT_PSSM, dim=1)
            ### feed fc + softmax for NEXT_AA path
            packed_out_NEXT_AA = self.fc_NEXT_AA(output.data)
            ##packed_out_NEXT_AA = F.softmax(packed_out_NEXT_AA, dim=1)
            return packed_out_NEXT_PSSM, packed_out_NEXT_AA, lstmp1_state, lstmp2_state
        elif self.predict_next_pssm:
            ### feed fc + softmax for NEXT_PSSM path
            packed_out_NEXT_PSSM = self.fc_NEXT_PSSM(output.data)
            packed_out_NEXT_PSSM = F.softmax(packed_out_NEXT_PSSM, dim=1)
            return packed_out_NEXT_PSSM, _, lstmp1_state, lstmp2_state
        else:
            ### feed fc + softmax for NEXT_AA path
            packed_out_NEXT_AA = self.fc_NEXT_AA(output.data)
            return _, packed_out_NEXT_AA, lstmp1_state, lstmp2_state


    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Best model found. Saving model... %s' % path)
        torch.save(self.state_dict(), path)
