from pathlib import Path
import numpy as np
import pickle
import torch

class Solver(object):
    default_adam_args = {"lr": 1e-2,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 1e-3}


    def __init__(self ,model,device_resource, optim=torch.optim.Adam, optim_args={}):
        """
            Inputs:
            - model: model object initialized from a torch.nn.Module
            - device_resource: Wether GPU or CPU
            - optim: optimizer used in training
            - optim_args: optimizer arguments
        """
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.device = device_resource
        self.model = model.to(self.device)
        self.optim = optim(self.model.parameters(), **self.optim_args)
        self.model = self.model.train()
        #self.pssm_loss_func = torch.nn.KLDivLoss(reduction='batchmean')
        self.aa_loss_func = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the loss.
        """
        self.train_loss_history_aa = []
        self.valset_loss_history_aa = []

    def _save__histories(self ,model_name):
        with open('../histories/' +model_name +'.data', 'wb') as filehandle:
            # store the data as binary data stream
            histories = {
                'train_loss_history_aa' :self.train_loss_history_aa,
                'valset_loss_history_aa' :self.valset_loss_history_aa,
            }
            pickle.dump(histories, filehandle)

    def compute_validation_loss(self, val_loader):
        mini_batch_val_loss_history_aa = []
        for j, (val_inputs, val_targets_aa) in enumerate(val_loader, 1):
            # ==> Process input regardless of the path
            #val_targets_pssm = val_targets_pssm * 0.01
            val_inputs = val_inputs.to(self.device).float()
            # get output from the model, given the inputs
            val_outputs_aa= self.model(val_inputs)
            ## ==> Process the targets
            #### For AA
            next_aa_loss = self.aa_loss_func(val_outputs_aa, val_targets_aa.to(self.device).long())
            mini_batch_val_loss_history_aa.append(next_aa_loss.data.cpu().numpy())
        # Compute approximate loss of the validation set
        self.valset_loss_history_aa.append(np.mean(mini_batch_val_loss_history_aa))

    def train(self, batch_id,train_loader, val_loader, tot_iters=0, processed_iters=0, best_val_loss=100,
              log_nth=1, model_name='Next_AA_Prediction'):
        """
        Inputs:
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - tot_iters: The total number of macro-iterations needed after all epochs are processed
        - processed_iters: Keep track of the previously processed macro-iterations in the whole training set
        - num_epochs: total number of training epochs
        - log_nth: log training and validation losses every nth iteration
        - model_name: name of the model
        """

        iter_in_current_batch = len(train_loader)

        if processed_iters == 0:
            ## Compute Validation Loss
            self.compute_validation_loss(val_loader)


        mini_batch_loss_history_aa = []

        for i, (inputs, targets_aa) in enumerate(train_loader, 1):
            # ==> Process input regardless of the path
            #targets_pssm = targets_pssm * 0.01
            # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
            self.optim.zero_grad()

            inputs = inputs.to(self.device).float()
            # get output from the model, given the inputs
            outputs_aa = self.model(inputs)
            ## ==> Process the targets
            next_aa_loss = 0
            next_aa_loss = self.aa_loss_func(outputs_aa, targets_aa.to(self.device).long())
            mini_batch_loss_history_aa.append(next_aa_loss.data.cpu().numpy())

            ## Backprop according to the path mode activated
            # get gradients w.r.t to parameters
            next_aa_loss.backward()
            self.optim.step()

            # Log the loss
            if (log_nth and i % log_nth == 0) or i == len(train_loader) or (processed_iters + i) == 1:
                if (processed_iters + i) == 1:
                    print('********************************************************************')
                    print('*******************   INITIAL LOG   ********************')
                    print('********************************************************************')
                    print('[Batch In Process: %d][Iteration %d/%d] aa loss => Train: %.3f / Val: %.3f ' % \
                          (batch_id,processed_iters + i,
                           tot_iters,
                           np.mean(mini_batch_loss_history_aa[-1]), self.valset_loss_history_aa[-1]))
                    print('********************************************************************')
                if (log_nth and i % log_nth == 0) or i == len(train_loader):
                    self.compute_validation_loss(val_loader)
                    self.train_loss_history_aa.append(np.mean(mini_batch_loss_history_aa))
                    print('[Batch In Process: %d][Iteration %d/%d] aa loss => Train: %.3f / Val: %.3f ' % \
                          (batch_id,processed_iters + i,
                           tot_iters,
                           self.train_loss_history_aa[-1], self.valset_loss_history_aa[-1]))
                    mini_batch_loss_history_aa = []
                    self._save__histories(model_name)
                    print('********************************************************************')

                ## Check if a better model is found and save
                thesis_model_dir = Path('../best_models')
                path = Path.joinpath(thesis_model_dir, model_name + '.pt')
                if best_val_loss > self.valset_loss_history_aa[-1]:
                    best_val_loss = self.valset_loss_history_aa[-1]
                    ##Save model
                    self.model.save(path)

        return best_val_loss, processed_iters + iter_in_current_batch