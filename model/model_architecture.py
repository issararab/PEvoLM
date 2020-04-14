import torch
import torch.nn as nn
import torch.nn.functional as F

class Inducer(torch.nn.Module):
    def __init__(self, hidden_size=512):
        super(Inducer, self).__init__()
        ############################################################################
        self.hidden_size = hidden_size
        self.fc_NEXT_AA = nn.Linear(hidden_size, 20, bias=True)
        # nn.init.xavier_uniform_(self.fc.weight)
        ############################################################################
    def forward(self, x):
        ############################################################################
        """
        The forward pass of the model architecture.

        Inputs:
        - x: a Tensor of size B x *
        """
        ############################################################################

        ### feed to fc
        out_NEXT_AA = self.fc_NEXT_AA(x)
        #packed_out_NEXT_PSSM = F.softmax(packed_out_NEXT_PSSM, dim=1)
        return out_NEXT_AA
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