from pathlib import Path
from allennlp.commands.elmo import ElmoEmbedder

##Load SeqVec
model_dir = Path('seq_vec_model')
weights = model_dir / 'weights.hdf5'
options = model_dir / 'options.json'
# embedder = ElmoEmbedder(options,weights, cuda_device=0)
embedder = ElmoEmbedder(options, weights, cuda_device=0)