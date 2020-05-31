import torch
import numpy as np
from model.lstm_with_projection import LSTMPLayer


class SeqEvoLM(torch.nn.Module):
    def __init__(self,device_resource, hidden_size=4096,projection_size=512,dropout_rate= 0.0):
        super(SeqEvoLM, self).__init__()
        self.device = device_resource
        self.hidden_size = hidden_size
        self.projection_size = projection_size
        ##Recurrent layers
        self.LSTMP_L1 = LSTMPLayer(512+20, self.hidden_size, self.projection_size,dropout_rate=dropout_rate)
        self.LSTMP_L2 = LSTMPLayer(512 * 2 + 256, self.hidden_size, self.projection_size,dropout_rate=dropout_rate)

    def forward(self, uncon_x, con_x, seq_lens):
        ############################################################################
        """
        The forward pass of the model architecture.

        Inputs:
        - uncon_x:  uncontextualized seqvec embedding of the sequences
                    a Tensor of size T x B x *, where B is the batch size and T the length of the longest sequence
        - con_x:  contextualized seqvec embedding of the sequences
                    a Tensor of size T x B x *, where B is the batch size and T the length of the longest sequence
        - seq_lens: a List of the original lengths of the sequences
        """
        ############################################################################

        uncon_x, con_x = uncon_x.to(self.device), con_x.to(self.device)
        ## feed to LSTM layers
        h1 = torch.zeros(uncon_x.size(1), self.projection_size, requires_grad=False).to(self.device)
        c1 = torch.zeros(uncon_x.size(1), self.hidden_size, requires_grad=False).to(self.device)
        h2 = torch.zeros(con_x.size(1), self.projection_size, requires_grad=False).to(self.device)
        c2 = torch.zeros(con_x.size(1), self.hidden_size, requires_grad=False).to(self.device)

        ##Architecture core
        ## pack_padded_seq giving it the input_size lengths
        #uncon_x_packed = pack_padded_sequence(uncon_x, seq_lens, batch_first=True, enforce_sorted=True)
        lstmp1_output, lstmp1_state = self.LSTMP_L1(uncon_x,[h1, c1])  # Shape (seq_length, batch_size, hidden_size)

        ##Concatinate with the contextualized SeqVec embedding
        #output = torch.cat((lstmp1_output,uncon_x, con_x), dim=2)
        output = torch.cat((lstmp1_output, con_x), dim=2)
        ##Run the second layer forward LSTM with projection
        lstmp2_output, lstmp2_state = self.LSTMP_L2(output, [h2, c2])  # Shape (seq_length, batch_size, hidden_size)

        ##Extract embeddings
        # Reshape the tensors to move the Batch size upfront
        uncon_x = uncon_x.permute(1,0,2).cpu()
        lstmp1_output = lstmp1_output.permute(1, 0, 2).cpu()
        lstmp2_output = lstmp2_output.permute(1,0,2).cpu()

        embeddings = []
        for i, embedding in enumerate(uncon_x,0):
            if i%2:
                uncon_embedding = torch.unsqueeze(embedding[1:seq_lens[i],:-20], dim=0)
                lstmp1_embedding = torch.unsqueeze(torch.cat((lstmp1_output[i,1:seq_lens[i],:],
                                                              torch.from_numpy(np.flip(
                                                                  lstmp1_output[i - 1, 1:seq_lens[i], :].detach().numpy(),
                                                                  0).copy())),dim=1),dim=0)

                lstmp2_embedding = torch.unsqueeze(torch.cat((lstmp2_output[i,1:seq_lens[i],:],
                                                              torch.from_numpy(np.flip(
                                                                  lstmp2_output[i - 1, 1:seq_lens[i], :].detach().numpy(),
                                                                  0).copy())),dim=1),dim=0)
                embeddings.append(torch.cat((uncon_embedding,lstmp1_embedding,lstmp2_embedding),dim=0))

        return embeddings # list of torch tensors (sequence embbedings) of shape (3 x L x 512)



    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Best model found. Saving model... %s' % path)
        torch.save(self.state_dict(), path)