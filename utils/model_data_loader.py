import numpy as np
import torch
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from utils.aa_encodings import  aa_class_labels_dic, aa_one_hot_encoding_dic
from utils.seq_vec_embedder import embedder, start_token_embed, end_token_embed
import gettext as _

def get_next_aa_labels(sequence):
    aa_labels = [aa_class_labels_dic.get(residue ,-1) for residue in sequence]
    return np.asarray(aa_labels, dtype=np.int64)

def get_sequence_aa_one_hot_encoding(sequence):
    sequence_aa_encoding = [aa_one_hot_encoding_dic.get(residue ,[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) for residue in sequence]
    return np.asarray(sequence_aa_encoding)


class MyDataset(data.Dataset):
    def __init__(self, list_headers ,list_sequences ,pssm_labels_dictionnary, predict_next_pssm = True, predict_next_aa = False):
        'Initialization'
        self.list_headers = list_headers
        self.list_sequences = list_sequences
        self.labels_dictionnary = pssm_labels_dictionnary
        self.predict_next_aa = predict_next_aa
        self.predict_next_pssm = predict_next_pssm

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_headers)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample/header
        header = self.list_headers[index]
        # retrieve sequence from list
        X = self.list_sequences[index]
        # Get PSSM labels
        #Y = self.labels_dictionnary[header]
        Y = np.concatenate((np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]),self.labels_dictionnary[header],np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])), axis=0)
        return X, Y, self.predict_next_pssm, self.predict_next_aa


def MyCollate(batch): #This function extracts SeqVec embedings, sequnce One-Hot encoding, and constructs both the next PSSM and AA prediction labels

    (xxx, yyy_pssm, predict_PSSM, predict_AA) = zip(*batch)
    ##Get embeddings
    ##Get list of lists
    seqs = [['<s>'] + list(xx) + ['<\s>'] for xx in xxx]
    ##Sort them by length and keep the indexes
    list_of_tuples = sorted(enumerate(seqs), key=lambda x: len(x[1]))

    ##Organize the sequences with their corresponding length order
    seq_length_threshold = 500 #Length threshold to extract SeqVec embeddings in parallel
    threshold_len_index = -1    #Seq index meeting the sequnce length threshold in the sorted list

    seqs = []
    pure_seqs = []
    seqs_one_hot_encoding = []
    pssm_labels = []
    for tupl in list_of_tuples:
        seqs.append(tupl[1])
        pure_seqs.append(tupl[1][1:-1])
        seqs_one_hot_encoding.append(get_sequence_aa_one_hot_encoding(tupl[1]))
        pssm_labels.append(yyy_pssm[tupl[0]])
        if len(tupl[1][1:-1]) <= seq_length_threshold:
            threshold_len_index = len(pure_seqs)

    del list_of_tuples, yyy_pssm, xxx

    if threshold_len_index != -1:
        if threshold_len_index == len(pure_seqs):
            xxx_embeddings = embedder.embed_sentences(pure_seqs)
        else:
            xxx_embeddings_temp = embedder.embed_sentences(pure_seqs[:threshold_len_index])
            xxx_embeddings = [embedding for embedding in xxx_embeddings_temp]
            xxx_embeddings += [embedder.embed_sentence(seq) for seq in pure_seqs[threshold_len_index:]]
    else:
        xxx_embeddings = [embedder.embed_sentence(seq) for seq in pure_seqs]

    # xxx_embeddings = embedder.embed_sentences(seqs)

    ##Split embedding tensors and concate with 1Hot encodings
    #expanded_xxx = []
    uncon_expanded_xxx = []
    con_expanded_xxx = []
    expanded_yyy_pssm = []
    expanded_yyy_aa = []
    xx_lens = []

    # Expanded SecVec embedding input, Expanded One-Hot encoding input, reshape, and concatinate
    for i,x_embedding in enumerate(xxx_embeddings):
        x_embedding = torch.FloatTensor(x_embedding).permute(1 ,0 ,2)
        x_embedding = torch.cat((start_token_embed,x_embedding,end_token_embed),dim=0)

        # Get uncontextualized forward and append
        forward_embedding = x_embedding[:-1 ,0,:512]
        forward_1hot_encoding = seqs_one_hot_encoding[i]
        forward_1hot_encoding = torch.from_numpy(forward_1hot_encoding[:-1, :])
        forward_in_uncon = torch.cat((forward_embedding, forward_1hot_encoding.float()), dim=1)
        uncon_expanded_xxx.append(forward_in_uncon)
        # Get contextualized forward and append
        forward_in_con = torch.cat((x_embedding[: ,1,:512],x_embedding[: ,2 ,:512]),dim = 1)
        forward_in_con = forward_in_con[:-1, :]
        con_expanded_xxx.append(forward_in_con)


        # Get uncontextualized backward, reverse and append
        backward_embedding = x_embedding[: ,:,512:].numpy()
        backward_embedding = np.flip(backward_embedding ,0)  # also [::-1,:,:] can be used
        backward_embedding_uncon = torch.from_numpy(backward_embedding[:-1 ,0,:].copy())
        backward_1hot_encoding = seqs_one_hot_encoding[i]
        backward_1hot_encoding = np.flip(backward_1hot_encoding, 0)  # [::-1,:,:]
        backward_1hot_encoding = torch.from_numpy(backward_1hot_encoding[:-1, :].copy())
        backward_in_uncon = torch.cat((backward_embedding_uncon, backward_1hot_encoding.float()), dim=1)
        uncon_expanded_xxx.append(backward_in_uncon)
        # Get contextualized backward and append
        backward_in_con = torch.cat((torch.from_numpy(backward_embedding[: ,1,:].copy()),torch.from_numpy(backward_embedding[: ,2,:].copy())), dim=1)
        backward_in_con = backward_in_con[:-1, :]
        con_expanded_xxx.append(backward_in_con)

        xx_lens.append(len(uncon_expanded_xxx[-1]))
        xx_lens.append(len(uncon_expanded_xxx[-1]))

    # Expanded ground truth
    yy_pad_pssm, yy_pad_aa = _ ,_
    if predict_PSSM[0]:
        # - PSSMs
        for y_pssm in pssm_labels:
            # Get forward and append
            expanded_yyy_pssm.append(torch.from_numpy(y_pssm[1: ,:]))
            # Get backward, reverse and append
            backward = np.flip(y_pssm ,0)  # also [::-1,:,:] can be used
            expanded_yyy_pssm.append(torch.from_numpy(backward[1: ,:].copy()))
        # Pad sequences
        yy_pad_pssm = pad_sequence(expanded_yyy_pssm[::-1])

    if predict_AA[0]:
        # - AAs
        for seq in seqs:
            # Get forward and append
            y_aa = get_next_aa_labels(seq)
            expanded_yyy_aa.append(torch.from_numpy(y_aa[1:]))
            # Get backward, reverse and append
            backward = np.flip(y_aa ,0  )  # y[::-1,:]
            expanded_yyy_aa.append(torch.from_numpy(backward[1:].copy()))
        # Pad sequences
        yy_pad_aa = pad_sequence(expanded_yyy_aa[::-1])
        del seqs

    # Reverse order of the batch to not affect the Truncated BPTT
    # as the batch is sorted in increasing order. -> Has to be sorted decreasing
    # Pad sequences
    uncon_xx_pad = pad_sequence(uncon_expanded_xxx[::-1])#, batch_first=True)
    con_xx_pad = pad_sequence(con_expanded_xxx[::-1])#, batch_first=True)
    xx_lens = xx_lens[::-1]

    return uncon_xx_pad, con_xx_pad, yy_pad_pssm, yy_pad_aa, xx_lens
