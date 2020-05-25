import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from utils.aa_encodings import  aa_one_hot_encoding_dic
from utils.seq_vec_embedder import embedder, start_token_embed, end_token_embed


def get_sequence_aa_one_hot_encoding(sequence):
    sequence_aa_encoding = [aa_one_hot_encoding_dic.get(residue ,[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) for residue in sequence]
    return np.asarray(sequence_aa_encoding)

def getSeqVecEmbeddings(batch,max_seq_length_parallel_embedding = 500): #This function extracts SeqVec embedings, sequnce One-Hot encoding, and constructs both the next PSSM and AA prediction labels
    """
        Get the SeqVec embedding to feed to SeqEvoLM

        Inputs:
        - batch: can be either a list of protein sequences or one sequence of type String
        - max_seq_length_parallel_embedding: sets the maximum length of the sequences in the batch
                                            to be processed in parallel by seqvec embedder

    """
    if type(batch) is list:
        xxx = batch
    else:
        xxx = [batch]

    ##Get embeddings
    ##Get list of lists
    seqs = [['<s>'] + list(xx) + ['<\s>'] for xx in xxx]
    ##Sort them by length and keep the indexes
    list_of_tuples = sorted(enumerate(seqs), key=lambda x: len(x[1]))

    ##Organize the sequences with their corresponding length order
    
    seq_length_threshold = max_seq_length_parallel_embedding #Length threshold to extract SeqVec embeddings in parallel
    threshold_len_index = -1    #Seq index meeting the sequence length threshold in the sorted list

    seqs = []
    seqs_one_hot_encoding = []
    original_indices = []
    for tupl in list_of_tuples:
        original_indices.append(tupl[0])
        seqs.append(tupl[1][1:-1])
        seqs_one_hot_encoding.append(get_sequence_aa_one_hot_encoding(tupl[1]))
        if len(tupl[1][1:-1]) <= seq_length_threshold:
            threshold_len_index = len(seqs)


    if threshold_len_index != -1 and len(batch) > 1:
        if threshold_len_index == len(seqs):
            xxx_embeddings = embedder.embed_sentences(seqs)
        else:
            xxx_embeddings_temp = embedder.embed_sentences(seqs[:threshold_len_index])
            xxx_embeddings = [embedding for embedding in xxx_embeddings_temp]
            xxx_embeddings += [embedder.embed_sentence(seq) for seq in seqs[threshold_len_index:]]
    else:
        xxx_embeddings = [embedder.embed_sentence(seq) for seq in seqs]

    ##Split embedding tensors and concate with 1Hot encodings
    uncon_expanded_xxx = []
    con_expanded_xxx = []
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

    # Reverse order of the batch to not affect the Truncated BPTT
    # as the batch is sorted in increasing order. -> Has to be sorted decreasing
    # Pad sequences
    uncon_xx_pad = pad_sequence(uncon_expanded_xxx[::-1])#, batch_first=True)
    con_xx_pad = pad_sequence(con_expanded_xxx[::-1])#, batch_first=True)
    xx_lens = xx_lens[::-1]
    original_indices = original_indices[::-1]

    return uncon_xx_pad, con_xx_pad, xx_lens, original_indices
