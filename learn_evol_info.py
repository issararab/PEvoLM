import sys
import pickle
import torch
from torch.utils import data
import math
import re
from model.model_architecture import Inducer
from model.model_solver import Solver
from utils.model_data_loader import MyDataset,MyCollate

train_config = {
    'predict_next_pssm': True,
    'predict_next_aa': True,
    'hidden_size': 256,
    'num_layers': 2,
    'epochs': 10,
    'max_seq_len_for_TBPTT': 300,
    'loss_harmony_weight': 0.5,
    'log_frequency': 1,
    'Learn_rate': 1e-3,
    'train_seq_path': '../inputs/train_set/',
    'train_label_path': '../labels/train_set/',
    'val_seq_path': '../inputs/val_set/',
    'val_label_path': '../labels/val_set/',
    'model_name': 'predict_next_pssm_and_AA_2Layers_RNN_32BatchSize_hidensize256',
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
    model = Inducer(device_resource=device,hidden_size=train_config['hidden_size'], num_layers=train_config['num_layers'],
                    predict_next_pssm=train_config['predict_next_pssm'],
                    predict_next_aa=train_config['predict_next_aa'])

    solver = Solver(model= model,max_seq_len_for_TBPTT=train_config['max_seq_len_for_TBPTT'], device_resource=device,
                    loss_harmony_weight=train_config['loss_harmony_weight'],
                    optim_args={"lr": train_config['Learn_rate']})

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
    input_10k_batches_to_process = 0
    for batch_name, batch_data in training_data.items():
        if input_10k_batches_to_process == 184:
            break
        tot_iters += math.ceil(len(batch_data['Headers']) / train_config['batch_size'])
        input_10k_batches_to_process += 1
    tot_iters *= train_config['epochs']
    input_10k_batches_to_process = 0
    ##Proceed to training
    for epoch in range(train_config['epochs']):
        input_10k_batches_to_process = 0
        for batch_name, batch_data in training_data.items():
            if input_10k_batches_to_process == 184:
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

            best_val_loss, processed_iters = solver.train(train_loader=train_loader, val_loader=val_loader,
                                                          tot_iters=tot_iters, processed_iters=processed_iters,
                                                          best_val_loss=best_val_loss, log_nth=train_config['log_frequency'],
                                                          model_name=train_config['model_name'])
            input_10k_batches_to_process += 1