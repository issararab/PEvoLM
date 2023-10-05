# PEvoLM
Protein Sequence Evolutionary Information Language Model

## Requirements

*  Python>=3.6.5
*  torch>=1.2.0
*  allennlp, version = 0.9

## seq_vec_model
The ELMo model trained on UniRef50 (=SeqVec) is available at:
[SeqVec-model](https://rostlab.org/~deepppi/seqvec.zip)

Please download the weights, copy the file, and paste it in the folder "seq_vec"

## PEvoLM ModelWeights
"seq_evo_lm" folder contains the pre-trained weights of this language model trained on 1.8 Million unique protein sequences. The model was trained on 2 GPUs of 16Gb of RAM each predicting both the next amino acid and the next PSSM column from both directions of a sequence (2-layer Bidirectional LSTMs with projections).

The embeddings are of size [3, L, 512], where L is the sequence length.
