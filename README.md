# Evolutionary Information Language Model
Using Variational Inference to Learn Representations of Protein Evolutionary Information.

This is a bi-directional language model embedding the protein sequence in a numerical representation encoding biophysical, biochemical, and the evolutionary information of the protein. The pretrained weights are availble in a sub-branch of the repo, 'Evol_Info_Embedder', trained on a set of 2 Million proteins in a cluster of GPUs granted by Google.

## Publication
If you use ANN-SoLo in your work, please cite the following publications:

- I. Arab, **PEvoLM: Protein Sequence Evolutionary Information Language Model**, 2023 IEEE Conference on Computational Intelligence in Bioinformatics and Computational Biology (CIBCB), Eindhoven, Netherlands, 2023, pp. 1-8, [doi:10.1109/CIBCB56990.2023.10264890](https://ieeexplore.ieee.org/document/10264890)

## Protein Sequence Evolutionary Information Language Model (PEvoLM)
![](img/LM_architecture.jpg?style=centerme)

## Requirements

*  Python>=3.6.5
*  torch>=1.2.0
*  allennlp v0.9

## seq_vec_model
The ELMo model trained on UniRef50 (=SeqVec) is available at:
[SeqVec-model](https://rostlab.org/~deepppi/seqvec.zip)

Please download the weights, copy the file, and paste it in the folder "seq_vec_model"


