import numpy as np
import torch
from torch import optim
from sklearn import metrics
from models.models import DataValueEstimater
import copy 

class Dvrl(object):
  """Data Valuation using Reinforcement Learning (DVRL) class.

    Attributes:
      x_train: training feature
      y_train: training labels
      x_valid: validation features
      y_valid: validation labels
      problem: 'regression' or 'classification'
      pred_model: predictive model (object)
      parameters: network parameters such as hidden_dim, iterations,
                  activation function, layer_number, learning rate
      checkpoint_file_name: File name for saving and loading the trained model
      flags: flag for training with stochastic gradient descent (flag_sgd)
             and flag for using pre-trained model (flag_pretrain)
  """

  def __init__(self, x_train, y_train, x_valid, y_valid,
               problem, pred_model, dve_train_param, predictor_train_parameters):
    self.x_train = x_train
    self.y_train = y_train
    self.x_valid = x_valid
    self.y_valid = y_valid
    self.epsilon = 1e-8

    self.problem = problem

    self.predictor_lr = predictor_train_parameters['lr']
    self.predictor_criterion = predictor_train_parameters['criterion']
    self.inner_iterations = predictor_train_parameters['iterations']
    self.batch_size_predictor = predictor_train_parameters['batch_size']

    self.dve_lr = dve_train_param['lr']
    self.outer_iterations = dve_train_param['iterations']
    self.batch_size = dve_train_param['batch_size']

    self.data_dim = len(x_train[0, :])
    self.label_dim = len(self.y_train_onehot[0, :])
    self.data_value_estimater = DataValueEstimater(x_dim=self.data_dim, y_dim=self.label_dim, y_hat_dim=self.label_dim)
    self.predictor = pred_model
    torch.save(self.predictor.state_dict(), 'tmp/pred_model')


    self.ori_model = copy.deepcopy(self.predictor)
    self.ori_model.load_state_dict(torch.load('tmp/pred_model'))
    self.ori_model = predictor_train(self.ori_model, 
                                self.x_train, 
                                self.y_train, 
                                batch_size=self.batch_size_predictor,
                                epochs=self.inner_iterations,
                                lr=self.predictor_lr,
                                criterion=self.predictor_criterion,
                                )
    
    self.val_model = copy.deepcopy(self.predictor)
    self.val_model.load_state_dict(torch.load('tmp/pred_model'))
    self.val_model = predictor_train(self.val_model, 
                                self.x_valid, 
                                self.y_valid, 
                                batch_size=self.batch_size_predictor,
                                epochs=self.inner_iterations,
                                lr=self.predictor_lr,
                                criterion=self.predictor_criterion,
                                )

    self.final_model = pred_model
    
    def dvrl_loss_fnc(selection_probs, s, reward):
       assert selection_probs.shape == torch.zeros(self.batch_size).shape, f"Expected shape [self.batch_size, 1], but got {selection_probs.shape}"
       assert s.shape == torch.zeros(self.batch_size).shape, f"Expected shape [self.batch_size], but got {s.shape}"
       prob = torch.sum(s * torch.log(selection_probs + self.epsilon) + (1 - s) * torch.log(1 - selection_probs + self.epsilon))
       return (-prob * reward) + \
        1e3 * (torch.max(torch.mean(selection_probs) - self.threshold, 0) + \
               torch.max((1 - self.threshold) - torch.mean(selection_probs), 0))

    def train_dvrl(self, perf_metric):
        y_valid_hat = self.ori_model(self.x_valid)
        assert len(y_valid_hat) == len(y_valid_hat.squeeze()), f"Shape Error at len(y_valid_hat) == len(y_valid_hat.squeeze())"
        y_valid_hat = y_valid_hat.squeeze()
        if perf_metric == 'auc':
            valid_perf = metrics.roc_auc_score(self.y_valid, y_valid_hat[:, 1], multi_class='ovr')

        elif perf_metric == 'accuracy':
            valid_perf = metrics.accuracy_score(self.y_valid, np.argmax(y_valid_hat,
                                                                    axis=1))
        elif perf_metric == 'log_loss':
            valid_perf = -metrics.log_loss(self.y_valid, y_valid_hat)
        elif perf_metric == 'rmspe':
            valid_perf = metrics.rmspe(self.y_valid, y_valid_hat)
        elif perf_metric == 'mae':
            valid_perf = metrics.mean_absolute_error(self.y_valid, y_valid_hat)
        elif perf_metric == 'mse':
            valid_perf = metrics.mean_squared_error(self.y_valid, y_valid_hat)
        
        y_train_valid_pred = self.val_model(self.x_train)
        if self.problem == 'classification':
            y_pred_diff = np.abs(self.y_train_onehot - y_train_valid_pred)
        elif self.problem == 'regression':
            y_pred_diff = \
                np.abs(self.y_train + 1e-5 - y_train_valid_pred)/(self.y_train + 1e-5)
            
        self.data_value_estimater.train()
        self.data_value_estimater = self.data_value_estimater.cuda()
        optimizer = optim.AdamW(self.data_value_estimater.parameters(), lr=self.dve_lr)
        
        for _ in self.outer_iterations:
            optimizer.zero_grad()
            batch_idx = np.random.permutation(len(self.x_train[:, 0]))[:self.batch_size]

            x_batch = self.x_train[batch_idx, :]
            y_batch = self.y_train[batch_idx]
            y_hat_batch = y_pred_diff[batch_idx]

            selection_probs = self.data_value_estimater(x_batch.cuda(), y_train.cuda(), y_hat_batch.cuda()).squeeze()
            sel_prob_curr = np.random.binomial(1, selection_probs, selection_probs.shape)

            if np.sum(sel_prob_curr) == 0:
                selection_probs = 0.5 * np.ones(np.shape(selection_probs))
                sel_prob_curr = np.random.binomial(1, selection_probs, selection_probs.shape)

            new_model = self.pred_model
            new_model.load_weights('tmp/pred_model')
            new_model = predictor_train(new_model, 
                                        x_batch, 
                                        y_batch, 
                                        sapmle_weight=sel_prob_curr,
                                        batch_size=self.batch_size_predictor,
                                        epochs=self.inner_iterations,
                                        criterion=self.predictor_criterion,
                                        lr=self.predictor_lr,
                                        )
            y_valid_hat = new_model(self.x_valid).squeeze()

            # Reward computation
            if perf_metric == 'auc':
                dvrl_perf = metrics.roc_auc_score(self.y_valid, y_valid_hat[:, 1])
            elif perf_metric == 'accuracy':
                dvrl_perf = metrics.accuracy_score(self.y_valid, np.argmax(y_valid_hat,
                                                                        axis=1))
            elif perf_metric == 'log_loss':
                dvrl_perf = -metrics.log_loss(self.y_valid, y_valid_hat)
            elif perf_metric == 'rmspe':
                dvrl_perf = metrics.rmspe(self.y_valid, y_valid_hat)
            elif perf_metric == 'mae':
                dvrl_perf = metrics.mean_absolute_error(self.y_valid, y_valid_hat)
            elif perf_metric == 'mse':
                dvrl_perf = metrics.mean_squared_error(self.y_valid, y_valid_hat)

            if self.problem == 'classification':
                reward_curr = dvrl_perf - valid_perf
            elif self.problem == 'regression':
                reward_curr = valid_perf - dvrl_perf

            loss = dvrl_loss_fnc(selection_probs, sel_prob_curr, reward_curr.cuda())
            loss.backward()
            optimizer.step()
            
        
        torch.save(self.data_value_estimater)
        selection_probs = self.data_value_estimater(self.x_train, self.y_train, y_pred_diff)[:, 0]
        self.final_model.load_weights('tmp/pred_model.h5')
        self.final_model = predictor_train(self.final_model,
                                        self.x_train,
                                        self.y_train,
                                        sample_weight=selection_probs,
                                        batch_size=self.batch_size_predictor,
                                        epochs=self.inner_iterations,
                                        criterion=self.predictor_criterion,
                                        lr=self.predictor_lr,
                                        )
 
    def predictor_train(model, x, y, batch_size, epochs, lr, criterion, sample_weights=None):
        # Assuming x and y_onehot are torch tensors
        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model = model.cuda()
        model.train()
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        for epoch in range(epochs):
            loss_all = 0
            for inputs, targets in dataloader:
                # Clear gradients
                optim.zero_grad()

                # Forward pass
                outputs = model(inputs.cuda())

                # Compute loss
                loss = criterion(outputs, targets.cuda(), weights=sample_weights)
                # Backward pass
                loss.backward()
                # Update weights
                optimizer.step()
                loss_all += loss.to('cpu').detach().numpy().copy()

            # Print loss for monitoring training progress
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss_all}')
        return model
        


    def data_valuator(self, x_train, y_train):
        if self.problem == 'classification':
            y_train_onehot = np.eye(len(np.unique(y_train)))[y_train.astype(int)]
            y_train_valid_pred = self.val_model.predict_proba(x_train)
        elif self.problem == 'regression':
            y_train_onehot = np.reshape(y_train, [len(y_train), 1])
            y_train_valid_pred = np.reshape(self.val_model.predict(x_train), [-1, 1])

        # Generates y_train_hat
        if self.problem == 'classification':
            y_train_hat = np.abs(y_train_onehot - y_train_valid_pred)
        elif self.problem == 'regression':
            y_train_hat = np.abs(y_train_onehot - y_train_valid_pred)/y_train_onehot

        dve = self.data_value_estimater()
        dve.load_state_dict('')

        selection_probs = dve(x_train, y_train_onehot, y_train_hat)

        return selection_probs
    
    def dvrl_predictor(self, x_test):
        return self.final_model.predict(x_test)


