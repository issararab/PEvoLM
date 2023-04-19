import gettext as _
import math
import math
import sys
from pathlib import Path

import numpy as np
import pickle
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


class Solver(object):
    default_adam_args = {"lr": 1e-2,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}


    def __init__(self ,model, max_seq_len_for_TBPTT, device_resource, loss_harmony_weight = 0.5, clip_norm_value = 10.0, optim=torch.optim.Adam, optim_args={}):
        """
            Inputs:
            - model: model object initialized from a torch.nn.Module
            - max_seq_len_for_TBPTT: the maximum length of sequences to process in one pass.
                                        In case of larger sequences, the mini-batch is sliced into chunks
                                        and processed via Truncated BPTT
            - device_resource: Wether GPU or CPU
            - loss_harmony_weight: a weighting factor used to balance the loss of multitask learning
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
        self.pssm_loss_func = torch.nn.KLDivLoss(reduction='batchmean')
        self.aa_loss_func = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
        self.max_seq_len_for_TBPTT = max_seq_len_for_TBPTT
        self.loss_harmony_weight = loss_harmony_weight
        self.clip_norm_value = clip_norm_value
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, 'min',
                                                                    patience=5,factor=0.5,min_lr=1e-3)
        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the loss.
        """
        self.mini_batch_train_loss_history_pssm = []
        self.mini_batch_loss_history_aa = []
        self.mini_batch_loss_history_combined = []
        self.train_time_step_indecies = [0]

        self.train_loss_history_pssm = []
        self.valset_loss_history_pssm = []
        self.train_loss_history_aa = []
        self.valset_loss_history_aa = []
        self.train_loss_history_combined = []
        self.valset_loss_history_combined = []

    def _save__histories(self ,model_name):
        with open('../histories/' +model_name +'.data', 'wb') as filehandle:
            # store the data as binary data stream
            histories = {
                'train_time_step_indecies': self.train_time_step_indecies,
                'per_mini_batch_train_loss_history_pssm':self.mini_batch_train_loss_history_pssm,
                'per_mini_batch_loss_history_aa':self.mini_batch_loss_history_aa,
                'per_mini_batch_loss_history_combined':self.mini_batch_loss_history_combined,
                'train_loss_history_pssm' :self.train_loss_history_pssm,
                'valset_loss_history_pssm' :self.valset_loss_history_pssm,
                'train_loss_history_aa' :self.train_loss_history_aa,
                'valset_loss_history_aa' :self.valset_loss_history_aa,
                'train_loss_history_combined' :self.train_loss_history_combined,
                'valset_loss_history_combined' :self.valset_loss_history_combined
            }
            pickle.dump(histories, filehandle)

    def tbptt_batches(self, inputs_uncon,inputs_con, seq_lens, pssm_targets, aa_targets, next_pssm, next_aa):
        max_seq_len = self.max_seq_len_for_TBPTT
        num_chunks = math.ceil(seq_lens[0] / max_seq_len)
        count_tot_processed_seqs = 0
        count_new_processed_seqs = 0
        max_seq_margin = 0.4
        for k in range(1 , num_chunks+2):
            new_seq_len = [max_seq_len if max_seq_len * k <= x else x% max_seq_len if (max_seq_len * (k - 1) + x % max_seq_len) == x else 0 for x in seq_lens]
            if k + 1 == num_chunks and seq_lens[0] % max_seq_len <= max_seq_len * max_seq_margin and seq_lens[0] % max_seq_len > 0:
                add_residues_to_training = [x % max_seq_len if (max_seq_len * (num_chunks - 1) + x % max_seq_len) == x else 0 for x in seq_lens]
                new_seq_len = np.add(new_seq_len, add_residues_to_training).tolist()
            #if new_seq_len[0] == max_seq_len and new_seq_len[4] == 0: #Computing resources based decision - depends on the batch size and the GPU memory
            #    if new_seq_len[3] != 0:                               #Condition block can be removed if only chunks of same size want to be processed
            #        new_seq_len[0] = new_seq_len[1] = seq_lens[0] - max_seq_len * (k-1)
            #        new_seq_len[2] = new_seq_len[3] = seq_lens[2] - max_seq_len * (k-1)
            #    else:
            #        new_seq_len[0] = new_seq_len[1] = seq_lens[0] - max_seq_len * (k-1)
            count_new_processed_seqs = new_seq_len.count(0) - count_tot_processed_seqs
            count_tot_processed_seqs = new_seq_len.count(0)
            if count_tot_processed_seqs:
                new_seq_len = list(filter(lambda a: a != 0, new_seq_len))
            chunked_inputs_uncon = inputs_uncon[max_seq_len * (k - 1):max_seq_len * (k - 1) + new_seq_len[0],
                                   : -count_tot_processed_seqs or None]
            chunked_inputs_con = inputs_con[max_seq_len * (k - 1):max_seq_len * (k - 1) + new_seq_len[0],
                                   : -count_tot_processed_seqs or None]
            if next_pssm and next_aa:
                chunked_targets_pssm = pssm_targets[max_seq_len * (k - 1):max_seq_len * (k - 1) + new_seq_len[0],
                                   : -count_tot_processed_seqs or None]
                chunked_targets_aa = aa_targets[max_seq_len * (k - 1):max_seq_len * (k - 1) + new_seq_len[0],
                                   : -count_tot_processed_seqs or None]
                yield chunked_inputs_uncon,chunked_inputs_con, chunked_targets_pssm, chunked_targets_aa, new_seq_len, count_new_processed_seqs
            elif next_pssm:
                chunked_targets_pssm = pssm_targets[max_seq_len * (k - 1):max_seq_len * (k - 1) + new_seq_len[0],
                                   : -count_tot_processed_seqs or None]
                yield chunked_inputs_uncon,chunked_inputs_con, chunked_targets_pssm, _, new_seq_len, count_new_processed_seqs
            else:
                chunked_targets_aa = aa_targets[max_seq_len * (k - 1):max_seq_len * (k - 1) + new_seq_len[0],
                                   : -count_tot_processed_seqs or None]
                yield chunked_inputs_uncon,chunked_inputs_con, _, chunked_targets_aa, new_seq_len, count_new_processed_seqs
            ## Break the loop in case of whole original batch fits the max_length threshold
            ## or the next few remaining residues in sequence is bellow the 30% max_length threshold
            if new_seq_len[0] > max_seq_len or k == num_chunks or num_chunks == 0:
                break

    def compute_validation_loss(self, val_loader):
        # Switch to evaluation mode
        self.model.eval()
        mini_batch_val_loss_history_pssm = []
        mini_batch_val_loss_history_aa = []
        mini_batch_val_loss_history_combined = []
        for j, (val_inputs_uncon, val_inputs_con, val_targets_pssm, val_targets_aa, val_seq_lens) in enumerate(val_loader, 1):
            val_inputs_uncon, val_inputs_con = val_inputs_uncon.cpu(), val_inputs_con.cpu()
            if self.model.predict_next_pssm:
                # Get probabilities for validation targets
                val_targets_pssm = val_targets_pssm * 0.01
            ##Starting parameters for Truncated BPTT to process validation set
            lstmp1_state, lstmp2_state = None, None
            time_steps_val_loss_history_pssm = []
            time_steps_val_loss_history_aa = []
            time_steps_val_loss_history_combined = []
            #print('Initial lengths list: {}'.format(val_seq_lens))
            for val_chunked_inputs_uncon,val_chunked_inputs_con, val_chunked_targets_pssm, val_chunked_targets_aa, new_seq_len, count_new_processed_seqs in self.tbptt_batches(
                    val_inputs_uncon,val_inputs_con, val_seq_lens, val_targets_pssm, val_targets_aa, self.model.predict_next_pssm,
                    self.model.predict_next_aa):
                #print('Chuncked lengths list: {}'.format(new_seq_len))
                # ==> Process input regardless of the path
                val_chunked_inputs_uncon, val_chunked_inputs_con = val_chunked_inputs_uncon.to(self.device).float(), val_chunked_inputs_con.to(self.device).float()
                # get output from the model, given the inputs
                val_outputs_pssm, val_outputs_aa, lstmp1_state, lstmp2_state = self.model(val_chunked_inputs_uncon, val_chunked_inputs_con, new_seq_len,
                                                                                          lstmp1_state, lstmp2_state, count_new_processed_seqs)

                ## ==> Process the targets
                combined_loss, next_pssm_loss, next_aa_loss = (0, 0, 0)
                #### For PSSM
                if self.model.predict_next_pssm:
                    val_chunked_targets_pssm_packed = pack_padded_sequence(val_chunked_targets_pssm, new_seq_len,
                                                                            enforce_sorted=True)

                    val_chunked_targets_pssm = val_chunked_targets_pssm_packed.data.to(self.device).float()
                    next_pssm_loss = self.pssm_loss_func(val_outputs_pssm.log(), val_chunked_targets_pssm)
                    time_steps_val_loss_history_pssm.append(next_pssm_loss.data.cpu().numpy())
                else:
                    time_steps_val_loss_history_pssm.append(next_pssm_loss)
                #### For AA
                if self.model.predict_next_aa:
                    val_chunked_targets_aa_packed = pack_padded_sequence(val_chunked_targets_aa, new_seq_len,
                                                                          enforce_sorted=True)

                    val_chunked_targets_aa = val_chunked_targets_aa_packed.data.to(self.device).float()
                    next_aa_loss = self.aa_loss_func(val_outputs_aa, val_chunked_targets_aa.long())
                    time_steps_val_loss_history_aa.append(next_aa_loss.data.cpu().numpy())
                else:
                    time_steps_val_loss_history_aa.append(next_aa_loss)
                #### For combined losses
                if self.model.predict_next_pssm and self.model.predict_next_aa:
                    combined_loss = self.loss_harmony_weight * next_pssm_loss + (
                                1 - self.loss_harmony_weight) * next_aa_loss
                    time_steps_val_loss_history_combined.append(combined_loss.data.cpu().numpy())
                else:
                    time_steps_val_loss_history_combined.append(combined_loss)
            mini_batch_val_loss_history_pssm.append(np.mean(time_steps_val_loss_history_pssm))
            mini_batch_val_loss_history_aa.append(np.mean(time_steps_val_loss_history_aa))
            mini_batch_val_loss_history_combined.append(np.mean(time_steps_val_loss_history_combined))
        # Compute approximate loss of the validation set
        self.valset_loss_history_combined.append(np.mean(mini_batch_val_loss_history_combined))
        self.valset_loss_history_pssm.append(np.mean(mini_batch_val_loss_history_pssm))
        self.valset_loss_history_aa.append(np.mean(mini_batch_val_loss_history_aa))
        self.model.train()

    def train(self, batch_id,train_loader, val_loader, tot_iters=0, processed_iters=0, best_val_loss=100,
              log_nth=1, model_name='Next_PSSM_Prediction'):
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

        mini_batch_loss_history_pssm = []
        mini_batch_loss_history_aa = []
        mini_batch_loss_history_combined = []
        for i, (inputs_uncon, inputs_con, targets_pssm, targets_aa, seq_lens) in enumerate(train_loader, 1):
            inputs_uncon, inputs_con = inputs_uncon.cpu(),inputs_con.cpu()
            if self.model.predict_next_pssm:
                # Get probabilities for training targets in case of predicting next pssm
                targets_pssm = targets_pssm * 0.01
            ### ====> New training
            ##Starting parameters for Truncated BPTT
            lstmp1_state, lstmp2_state = None, None
            time_steps_loss_history_pssm = []
            time_steps_loss_history_aa = []
            time_steps_loss_history_combined = []
            #print('Initial lengths list: {}'.format(seq_lens))
            for chunked_inputs_uncon, chunked_inputs_con, chunked_targets_pssm, chunked_targets_aa, new_seq_len, count_new_processed_seqs in self.tbptt_batches(
                    inputs_uncon, inputs_con, seq_lens, targets_pssm, targets_aa, self.model.predict_next_pssm, self.model.predict_next_aa):
                #print('Chuncked lengths list: {}'.format(new_seq_len))
                # ==> Process input regardless of the path
                # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
                self.optim.zero_grad()

                chunked_inputs_uncon, chunked_inputs_con = chunked_inputs_uncon.to(self.device).float(), chunked_inputs_con.to(self.device).float()
                # get output from the model, given the inputs
                outputs_pssm, outputs_aa, lstmp1_state, lstmp2_state = self.model(chunked_inputs_uncon, chunked_inputs_con, new_seq_len,
                                                                                  lstmp1_state, lstmp2_state, count_new_processed_seqs)
                ## ==> Process the targets
                combined_loss, next_pssm_loss, next_aa_loss = (0, 0, 0)
                #### For PSSM
                if self.model.predict_next_pssm:
                    chunked_targets_pssm_packed = pack_padded_sequence(chunked_targets_pssm, new_seq_len,
                                                                        enforce_sorted=True)
                    chunked_targets_pssm = chunked_targets_pssm_packed.data.to(self.device).float()
                    next_pssm_loss = self.pssm_loss_func(outputs_pssm.log(), chunked_targets_pssm)
                    time_steps_loss_history_pssm.append(next_pssm_loss.data.cpu().numpy())
                else:
                    time_steps_loss_history_pssm.append(next_pssm_loss)
                #### For aa
                if self.model.predict_next_aa:
                    chunked_targets_aa_packed = pack_padded_sequence(chunked_targets_aa, new_seq_len,
                                                                      enforce_sorted=True)
                    chunked_targets_aa = chunked_targets_aa_packed.data.to(self.device).float()
                    next_aa_loss = self.aa_loss_func(outputs_aa, chunked_targets_aa.long())
                    time_steps_loss_history_aa.append(next_aa_loss.data.cpu().numpy())
                else:
                    time_steps_loss_history_aa.append(next_aa_loss)
                #### For combined losses
                if self.model.predict_next_pssm and self.model.predict_next_aa:
                    combined_loss = self.loss_harmony_weight * next_pssm_loss + (
                                1 - self.loss_harmony_weight) * next_aa_loss
                    time_steps_loss_history_combined.append(combined_loss.data.cpu().numpy())
                else:
                    time_steps_loss_history_combined.append(combined_loss)

                ## Backprop according to the path mode activated
                # get gradients w.r.t to parameters
                if self.model.predict_next_pssm and self.model.predict_next_aa:
                    # print('Both   ---- BACKPROP')
                    combined_loss.backward()
                elif self.model.predict_next_aa:
                    # print('AA     ----  BACKPROP')
                    next_aa_loss.backward()
                else:
                    # print('PSSM     ----  BACKPROP')
                    next_pssm_loss.backward()
                ##Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm_value)
                self.optim.step()

            mini_batch_loss_history_combined.append(np.mean(time_steps_loss_history_combined))
            mini_batch_loss_history_pssm.append(np.mean(time_steps_loss_history_pssm))
            mini_batch_loss_history_aa.append(np.mean(time_steps_loss_history_aa))
            # Log the loss
            if (log_nth and i % log_nth == 0) or i == len(train_loader) or (processed_iters + i) == 1:
                ## Case - first log
                if (processed_iters + i) == 1:
                    self.train_loss_history_combined.append(mini_batch_loss_history_combined[-1])
                    self.train_loss_history_pssm.append(mini_batch_loss_history_pssm[-1])
                    self.train_loss_history_aa.append(mini_batch_loss_history_aa[-1])
                    print('********************************************************************')
                    print('*******************   INITIAL VALIDATION LOSS   ********************')
                    print('********************************************************************')
                    if self.model.predict_next_pssm and self.model.predict_next_aa:
                        print('[Batch In Process: %d][Iteration %d/%d] combined loss => Train: %.3f / Val: %.3f ' % \
                              (batch_id, processed_iters + i,
                               tot_iters,
                               self.train_loss_history_combined[-1], self.valset_loss_history_combined[-1]))
                    if self.model.predict_next_aa:
                        print('[Batch In Process: %d][Iteration %d/%d] aa loss => Train: %.3f / Val: %.3f ' % \
                              (batch_id, processed_iters + i,
                               tot_iters,
                               self.train_loss_history_aa[-1], self.valset_loss_history_aa[-1]))
                    if self.model.predict_next_pssm:
                        print('[Batch In Process: %d][Iteration %d/%d] pssm loss => Train: %.3f / Val: %.3f ' % \
                              (batch_id, processed_iters + i,
                               tot_iters,
                               self.train_loss_history_pssm[-1], self.valset_loss_history_pssm[-1]))
                    print('********************************************************************')

                ## Case - meeting log criterion
                if (log_nth and i % log_nth == 0) or i == len(train_loader):
                    self.mini_batch_train_loss_history_pssm += mini_batch_loss_history_pssm
                    self.mini_batch_loss_history_aa += mini_batch_loss_history_aa
                    self.mini_batch_loss_history_combined += mini_batch_loss_history_combined
                    self.train_loss_history_combined.append(np.mean(mini_batch_loss_history_combined))
                    self.train_loss_history_pssm.append(np.mean(mini_batch_loss_history_pssm))
                    self.train_loss_history_aa.append(np.mean(mini_batch_loss_history_aa))
                    if i == int(int(len(train_loader)/log_nth)/2) * log_nth:
                        self.compute_validation_loss(val_loader)
                    if self.model.predict_next_pssm and self.model.predict_next_aa:
                        self.scheduler.step(self.valset_loss_history_combined[-1])
                    elif self.model.predict_next_pssm:
                        self.scheduler.step(self.valset_loss_history_pssm[-1])
                    else:
                        self.scheduler.step(self.valset_loss_history_aa[-1])


                    if self.model.predict_next_pssm and self.model.predict_next_aa:
                        print('[Batch In Process: %d][Iteration %d/%d] COMBINED loss => Train: %.3f / Val: %.3f ' % \
                              (batch_id,processed_iters + i,
                               tot_iters,
                               self.train_loss_history_combined[-1], self.valset_loss_history_combined[-1]))
                    if self.model.predict_next_pssm:
                        print('[Batch In Process: %d][Iteration %d/%d] PSSM loss => Train: %.3f / Val: %.3f ' % \
                              (batch_id,processed_iters + i,
                               tot_iters,
                               self.train_loss_history_pssm[-1], self.valset_loss_history_pssm[-1]))
                    if self.model.predict_next_aa:
                        print('[Batch In Process: %d][Iteration %d/%d] AA loss => Train: %.3f / Val: %.3f ' % \
                              (batch_id,processed_iters + i,
                               tot_iters,
                               self.train_loss_history_aa[-1], self.valset_loss_history_aa[-1]))
                    if math.isnan(self.train_loss_history_aa[-1]):
                        thesis_model_dir = Path('../best_models')
                        path = Path.joinpath(thesis_model_dir, model_name + '_nan.pt')
                        self.model.save(path)
                        sys.exit(0)
                    print('********************************************************************')
                    # Save the histories
                    self._save__histories(model_name)
                    mini_batch_loss_history_pssm = []
                    mini_batch_loss_history_aa = []
                    mini_batch_loss_history_combined = []

                ## Check if a better model is found and save
                thesis_model_dir = Path('../best_models')
                path = Path.joinpath(thesis_model_dir, model_name + '.pt')
                if self.model.predict_next_pssm and self.model.predict_next_aa:
                    if best_val_loss > self.valset_loss_history_combined[-1]:
                        best_val_loss = self.valset_loss_history_combined[-1]
                        ##Save model
                        self.model.save(path)
                elif self.model.predict_next_aa:
                    if best_val_loss > self.valset_loss_history_aa[-1]:
                        best_val_loss = self.valset_loss_history_aa[-1]
                        ##Save model
                        self.model.save(path)
                else:
                    if best_val_loss > self.valset_loss_history_pssm[-1]:
                        best_val_loss = self.valset_loss_history_pssm[-1]
                        ##Save model
                        self.model.save(path)

        return best_val_loss, processed_iters + iter_in_current_batch
