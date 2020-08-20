# Evol_Info_Prediction
Variational Inference to Learn Representations of Protein Evolutionary Information.

This is a bi-directional language model embedding the protein sequence in a numerical representation incorporating biophysical, biochemical, and evolutionary information. The pretrained weights are availble in a sub-branch of the repo, 'Evol_Info_Embedder', trained on a set of 2 Million proteins in a cluster of GPUs granted by Google.

## Requirements

*  Python>=3.6.5
*  torch>=1.2.0
*  allennlp

## seq_vec_model
The ELMo model trained on UniRef50 (=SeqVec) is available at:
[SeqVec-model](https://rostlab.org/~deepppi/seqvec.zip)

Please download the weights, copy the file, and paste it in the folder "seq_vec_model"

## Sequence Evolution Language Model embedder (Seq_Evo_LM)
![](img/LM_with_residual.jpg?style=centerme)
