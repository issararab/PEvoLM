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
    'hidden_size': 1024,
    'epochs': 10,
    'log_frequency': 30,
    'Learn_rate': 1e-3,
    'train_seq_path': '../inputs/train_set/',
    'train_label_path': '../labels/train_set/',
    'val_seq_path': '../inputs/val_set/',
    'val_label_path': '../labels/val_set/',
    #'train_seq_path': '../inputs/test_set/',
    #'train_label_path': '../labels/test_set/',
    #'val_seq_path': '../inputs/val_set/',
    #'val_label_path': '../labels/val_set/',
    'model_name': 'predict_next_AA_from_SeqVec_last_layer_32BatchSize',
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
    model = Inducer(hidden_size=train_config['hidden_size'])

    solver = Solver(model= model, device_resource=device, optim_args={"lr": train_config['Learn_rate']})

    # Load training sequences
    with open(train_config['train_seq_path'] + 'training_set.data', 'rb') as filehandle:
        # read the data as binary data stream
        training_data = pickle.load(filehandle)

    # Load validation sequences
    with open(train_config['val_seq_path'] + 'val_set.data', 'rb') as filehandle:
        # read the data as binary data stream
        validation_data = pickle.load(filehandle)

    val_dataset = MyDataset(validation_data['mini_batch_1']['Sequences'])
    val_loader = data.DataLoader(dataset=val_dataset,
                                 batch_size=train_config['batch_size'],
                                 shuffle=True,
                                 collate_fn=MyCollate)

    best_val_loss = sys.maxsize
    processed_iters = 0
    ##Count total number of training iterations
    tot_iters = 0
    for batch_id, (batch_name, batch_data) in enumerate(training_data.items(), 1):
        if batch_id == 10:
            break
        tot_iters += math.ceil(len(batch_data['Headers']) / train_config['batch_size'])
    tot_iters *= train_config['epochs']

    print('START TRAIN.')
    ##Proceed to training
    for epoch in range(train_config['epochs']):
        for batch_id, (batch_name, batch_data) in enumerate(training_data.items(), 1):
            if batch_id == 10:
                break

            train_dataset = MyDataset(batch_data['Sequences'])
            train_loader = data.DataLoader(dataset=train_dataset,
                                           batch_size=train_config['batch_size'],
                                           shuffle=True,
                                           collate_fn=MyCollate)

            best_val_loss, processed_iters = solver.train(batch_id,train_loader=train_loader, val_loader=val_loader,
                                                          tot_iters=tot_iters, processed_iters=processed_iters,
                                                          best_val_loss=best_val_loss, log_nth=train_config['log_frequency'],
                                                          model_name=train_config['model_name'])
    print('FINISH.')