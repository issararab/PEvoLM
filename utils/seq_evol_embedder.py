import os
from pathlib import Path
import torch
from utils.seq_evo_lm_data_loader import getSeqVecEmbeddings
from model.embedder_architecture import SeqEvoLM
import json

def SeqEvoLmEmbedder(options):
    with open(options) as json_file:
        options = json.load(json_file)
        if options['use_gpu']:
            ##Check for GPU
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SeqEvoLM(device_resource=device,hidden_size=options['lstmp']['hidden_size'],projection_size=options['lstmp']['projection_size'],dropout_rate=options['lstmp']['dropout_rate'])
        ### Load saved model weights
        thesis_model_dir = Path(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'seq_evo_lm')))
        path = Path.joinpath(thesis_model_dir, options['model_name']+ '.pt')
        model.load_state_dict(torch.load(path, map_location=device))
        model = model.to(device)
        model.eval()
        return model

def getSeqEvoLmEmbeddings(SeqEvoLmEmbedder,batch): #Function takes one string or a list of strings passed to the argument 'batch'
    uncon_xx_pad, con_xx_pad, xx_lens, original_indices = getSeqVecEmbeddings(batch)
    embeddings = SeqEvoLmEmbedder(uncon_xx_pad, con_xx_pad, xx_lens)
    embeddings = [embeddings[i] for i in original_indices]
    return embeddings

model_dir = Path(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'seq_evo_lm')))
options = Path.joinpath(model_dir,'options.json')
embedder = SeqEvoLmEmbedder(options)
