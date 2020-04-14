import numpy as np
import torch
from torch.utils import data
from utils.aa_encodings import  aa_class_labels_dic, aa_one_hot_encoding_dic
from utils.seq_vec_embedder import embedder
import gettext as _

def get_next_aa_labels(sequence):
    aa_labels = [aa_class_labels_dic.get(residue ,-1) for residue in sequence]
    return np.asarray(aa_labels, dtype=np.int64)

class MyDataset(data.Dataset):
    def __init__(self, list_sequences):
        'Initialization'
        self.list_sequences = list_sequences

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_sequences)

    def __getitem__(self, index):
        'Generates one sample of data'
        # retrieve sequence from list
        X = self.list_sequences[index]
        return X


def MyCollate(batch): #This function extracts SeqVec embedings, sequnce One-Hot encoding, and constructs both the next PSSM and AA prediction labels


    xxx = batch
    ##Get embeddings
    ##Get list of lists
    seqs = [['<s>'] + list(xx) + ['<\s>'] for xx in xxx]
    ##Sort them by length and keep the indexes
    list_of_tuples = sorted(enumerate(seqs), key=lambda x: len(x[1]))

    ##Organize the sequences with their corresponding length order
    seq_length_threshold = 1000 #Length threshold to extract SeqVec embeddings in parallel
    threshold_len_index = -1    #Seq index meeting the sequnce length threshold in the sorted list

    seqs = []
    for tupl in list_of_tuples:
        seqs.append(tupl[1])
        if len(tupl[1]) <= seq_length_threshold:
            threshold_len_index = len(seqs)

    if threshold_len_index != -1:
        if threshold_len_index == len(seqs):
            xxx_embeddings = embedder.embed_sentences(seqs)
        else:
            xxx_embeddings_temp = embedder.embed_sentences(seqs[:threshold_len_index])
            xxx_embeddings = [embedding for embedding in xxx_embeddings_temp]
            xxx_embeddings += [embedder.embed_sentence(seq) for seq in seqs[threshold_len_index:]]
    else:
        xxx_embeddings = [embedder.embed_sentence(seq) for seq in seqs]



    expanded_xxx = []
    expanded_yyy_aa = []
    #expanded_yyy_pssm = []
    # SecVec embedding input
    for x_embedding in xxx_embeddings:
        x_embedding = torch.FloatTensor(x_embedding[-1,:,:])
        #x_embedding = torch.FloatTensor(x_embedding[-1,:,:512]) ## In case of forward embedding
        # x_embedding = torch.FloatTensor(x_embedding[-1,:,512:]) ## In case of backward embedding
        expanded_xxx.append(x_embedding[:-1,:])
    expanded_xxx= torch.cat(expanded_xxx, dim=0)

    # - AAs
    for seq in seqs:
        # Get forward and append
        y_aa = get_next_aa_labels(seq)
        expanded_yyy_aa.append(torch.from_numpy(y_aa[1:]))
    expanded_yyy_aa = torch.cat(expanded_yyy_aa, dim=0)

    """
    # - PSSMs
    for y_pssm in pssm_labels:
        # Get forward and append
        expanded_yyy_pssm.append(torch.from_numpy(y_pssm[1:, :]))
    expanded_yyy_pssm = torch.cat(expanded_yyy_pssm, dim=0)
    """
    return expanded_xxx, expanded_yyy_aa #expanded_yyy_pssm