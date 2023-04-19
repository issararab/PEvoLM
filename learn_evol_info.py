import math
import pickle
import re
import sys
from pathlib import Path

import torch
from torch.utils import data

from model.model_architecture import Inducer
from utils.model_data_loader import MyDataset,MyCollate
from model.model_solver import Solver


train_config = {
    'predict_next_pssm': True,
    'predict_next_aa': True,
    'hidden_size': 2048,
    'projection_size': 256,
    'dropout_rate': 0.1,
    'epochs': 1,
    'max_seq_len_for_TBPTT': 290,
    'loss_harmony_weight': 0.75,
    'log_frequency': 15,
    'Learn_rate': 0.02,
    'clip_norm_value': 1.0,
    'train_seq_path': '../inputs/train_set/',
    'train_label_path': '../labels/train_set/',
    'val_seq_path': '../inputs/val_set/',
    'val_label_path': '../labels/val_set/',
    'model_name': 'predict_next_PSSM_and_AA_2Layers_32BatchSize_hidenSize2048_projectionSize256_TBPTT300_Cliping_0.1Dropout_residual',#predict_next_pssm_and_AA
    'batch_size': 32

}


if __name__ == "__main__":

    ##Parse the training arguments
    args_len = len(sys.argv) - 1
    if args_len % 2 != 0:
        print("Error in command! Pleae respect format.")
        sys.exit(0)
    else:
        arg_index = 1
        while arg_index < args_len:
            if train_config.get(sys.argv[arg_index][2:],False):
                train_config[sys.argv[arg_index][2:]] = int(sys.argv[arg_index + 1]) if sys.argv[arg_index + 1].isnumeric() \
                    else float(sys.argv[arg_index + 1]) if re.match(r'^[+-]?\d(>?\.\d+)?$', sys.argv[arg_index + 1]) \
                    else True if sys.argv[arg_index + 1] == 'True' \
                    else False if sys.argv[arg_index + 1] == 'False' \
                    else sys.argv[arg_index + 1]
                arg_index += 2
            else:
                print('Argument < {} > does not exist. Pleae respect format.'.format(sys.argv[arg_index]))
                sys.exit(0)

    ##Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ## Instantiate the model and the solver
    model = Inducer(device_resource=device,hidden_size=train_config['hidden_size'],projection_size=train_config['projection_size'],
                    dropout_rate=train_config['dropout_rate'], predict_next_pssm=train_config['predict_next_pssm'],
                    predict_next_aa=train_config['predict_next_aa'])

    ### Load saved model weights
    thesis_model_dir = Path('../best_models')
    path = Path.joinpath(thesis_model_dir,
                         'predict_next_PSSM_and_AA_2Layers_32BatchSize_hidenSize2048_projectionSize256_TBPTT300_Cliping_0.1Dropout_residual_' + '.pt')
    #predict_next_PSSM_2Layers_32BatchSize_hidenSize2048_projectionSize256_TBPTT300_Cliping_0.1Dropout
    #path = Path.joinpath(thesis_model_dir, 'predict_next_AA_PSSM_2Layers_16BatchSize_hidenSize4096_projectionSize256_TBPTT300_Cliping_0.1Dropout' + '.pt')
    model.load_state_dict(torch.load(path))
    ### Load saved model weights

    solver = Solver(model = torch.nn.DataParallel(model),max_seq_len_for_TBPTT=train_config['max_seq_len_for_TBPTT'],
                    device_resource=device,loss_harmony_weight=train_config['loss_harmony_weight'],
                    clip_norm_value = train_config['clip_norm_value'],
                    optim_args={"lr": train_config['Learn_rate']})

    #Print total number of trainable prameters
    print('Total number of the model parameters: {}'.format(sum(p.numel() for p in model.parameters())))
    print('Total number of trainable the model parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    # Load training sequences
    with open(train_config['train_seq_path'] + 'training_set.data', 'rb') as filehandle:
        # read the data as binary data stream
        training_data = pickle.load(filehandle)

    # Load validation sequences
    with open(train_config['val_seq_path'] + 'val_set.data', 'rb') as filehandle:
        # read the data as binary data stream
        validation_data = pickle.load(filehandle)
    # Load pssm validation labels
    with open(train_config['val_label_path'] + 'mini_batch_1_labels.data', 'rb') as filehandle:
        # read the data as binary data stream
        validation_labels = pickle.load(filehandle)

    val_dataset = MyDataset(validation_data['mini_batch_1']['Headers'],
                            validation_data['mini_batch_1']['Sequences'], validation_labels,
                            predict_next_pssm=train_config['predict_next_pssm'],
                            predict_next_aa=train_config['predict_next_aa'])
    val_loader = data.DataLoader(dataset=val_dataset,
                                 batch_size=train_config['batch_size'],
                                 shuffle=True,
                                 collate_fn=MyCollate)

    best_val_loss = sys.maxsize
    processed_iters = 0
    ##Count total number of training iterations
    tot_iters = 0
    for batch_id, (batch_name, batch_data) in enumerate(training_data.items(), 1):
        if batch_id < 13:
            continue
        if batch_id == 21:
            break
        tot_iters += math.ceil(len(batch_data['Headers']) / train_config['batch_size'])
    tot_iters *= train_config['epochs']
    ##Proceed to training
    if torch.cuda.device_count() > 1:
        print("Let's use ", torch.cuda.device_count() ," GPUs!")
    print('START TRAIN.')
    for epoch in range(train_config['epochs']):
        for batch_id,(batch_name, batch_data) in enumerate(training_data.items(),1):
            if batch_id < 13:
                continue
            if batch_id == 21:
                break
            # Load pssm training labels
            with open(train_config['train_label_path'] + batch_name + '_labels.data', 'rb') as filehandle:
                # read the data as binary data stream
                training_labels = pickle.load(filehandle)

            train_dataset = MyDataset(batch_data['Headers'], batch_data['Sequences'], training_labels,
                                      predict_next_pssm=train_config['predict_next_pssm'],
                                      predict_next_aa=train_config['predict_next_aa'])

            train_loader = data.DataLoader(dataset=train_dataset,
                                           batch_size=train_config['batch_size'],
                                           shuffle=True,
                                           collate_fn=MyCollate)

            best_val_loss, processed_iters = solver.train(batch_id,train_loader=train_loader, val_loader=val_loader,
                                                          tot_iters=tot_iters, processed_iters=processed_iters,
                                                          best_val_loss=best_val_loss, log_nth=train_config['log_frequency'],
                                                          model_name=train_config['model_name'])
    print('FINISH.')
