# Evolutionary Information Language Model
Using Variational Inference to Learn Representations of Protein Evolutionary Information.

This is a bi-directional language model embedding the protein sequence in a numerical representation encoding biophysical, biochemical, and the evolutionary information of the protein. The pretrained weights are availble in a sub-branch of the repo, 'Evol_Info_Embedder', trained on a set of 2 Million proteins in a cluster of GPUs granted by Google.

## Protein Sequence Evolutionary Information Language Model (PEvoLM)
![](img/LM_with_residual.jpg?style=centerme)

## Requirements

*  Python>=3.6.5
*  torch>=1.2.0
*  allennlp v0.9

## seq_vec_model
The ELMo model trained on UniRef50 (=SeqVec) is available at:
[SeqVec-model](https://rostlab.org/~deepppi/seqvec.zip)

Please download the weights, copy the file, and paste it in the folder "seq_vec_model"


