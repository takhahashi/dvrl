import hydra
import numpy as np
import keras
import tensorflow as tf
import pandas as pd
import lightgbm
from transformers import AutoTokenizer, AutoModel
from utils.dataset import get_upper_score, get_dataset
import json
# Sets current directory
# os.chdir(repo_dir)
import sys
from omegaconf import DictConfig
import torch

from utils.loss_fnc import weighted_mse
from data_loading import load_tabular_data, preprocess_data
import dvrl_torch
from dvrl_metrics import remove_high_low
from models.models import Predictor

def simple_collate_fn(list_of_data):
  pad_max_len = torch.tensor(0)
  for data in list_of_data:
    if(torch.count_nonzero(data['attention_mask']) > pad_max_len):
      pad_max_len = torch.count_nonzero(data['attention_mask'])
  in_ids, token_type, atten_mask, labels = [], [], [], []
  for data in list_of_data:
    in_ids.append(data['input_ids'][:pad_max_len])
    token_type.append(data['token_type_ids'][:pad_max_len])
    atten_mask.append(data['attention_mask'][:pad_max_len])
    labels.append(data['labels'])
  batched_tensor = {}
  batched_tensor['input_ids'] = torch.stack(in_ids)
  batched_tensor['token_type_ids'] = torch.stack(token_type)
  batched_tensor['attention_mask'] = torch.stack(atten_mask)
  batched_tensor['labels'] = torch.tensor(labels)
  return batched_tensor


def extract_clsvec(bert_model, dataloader):
    bert_model = bert_model.cuda()
    bert_model.eval()
    eval_results = {}
    for t_data in dataloader:
        batch = {k: v.cuda() for k, v in t_data.items()}
        y_true = {'labels': batch['labels'].to('cpu').detach().numpy().copy()}
        x = {'input_ids':batch['input_ids'],
                    'attention_mask':batch['attention_mask'],
                    'token_type_ids':batch['token_type_ids']}

        cls_outputs = {k: v.to('cpu').detach().numpy().copy() for k, v in bert_model(x)['last_hidden_state'][:, 0, :]}

        if len(eval_results) == 0:
            eval_results.update(cls_outputs)
            eval_results.update(y_true)
        else:
            cls_outputs.update(y_true)
            eval_results = {k1: np.concatenate([v1, v2]) for (k1, v1), (k2, v2) in zip(eval_results.items(), cls_outputs.items())}
        
    return eval_results['hidden_state'], eval_results['labels']


@hydra.main(config_path="/content/drive/MyDrive/GoogleColab/dvrl/configs", config_name="train_reg")
def main(cfg: DictConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)
    bert = AutoModel.from_pretrained(cfg.model.model_name_or_path)
    upper_score = get_upper_score(cfg.sas.question_id, cfg.sas.score_id)

    with open(cfg.path.traindata_file_name) as f:
        train_dataf = json.load(f)
    train_dataset = get_dataset(train_dataf, cfg.sas.score_id, upper_score, cfg.model.reg_or_class, tokenizer)
    with open(cfg.path.valdata_file_name) as f:
        dev_dataf = json.load(f)
    dev_dataset = get_dataset(dev_dataf, cfg.sas.score_id, upper_score, cfg.model.reg_or_class, tokenizer)
    with open(cfg.path.traindata_file_name) as f:
        train_dataf = json.load(f)
    train_dataset = get_dataset(train_dataf, cfg.sas.score_id, upper_score, cfg.model.reg_or_class, tokenizer)
    #with open(cfg.path.testdata_file_name) as f:
    #    test_dataf = json.load(f)
    #test_dataset = get_dataset(test_dataf, cfg.sas.score_id, upper_score, cfg.model.reg_or_class, tokenizer)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                    batch_size=cfg.training.batch_size, 
                                                    shuffle=True, 
                                                    drop_last=False, 
                                                    collate_fn=simple_collate_fn)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, 
                                                batch_size=cfg.training.batch_size, 
                                                shuffle=False, 
                                                drop_last=False, 
                                                collate_fn=simple_collate_fn)
    #test_dataloader = torch.utils.data.DataLoader(dev_dataset, 
    #                                            batch_size=cfg.training.batch_size, 
    #                                            shuffle=False, 
    #                                            drop_last=False, 
    #                                            collate_fn=simple_collate_fn)

    x_train, y_train = extract_clsvec(bert, train_dataloader)
    x_valid, y_valid = extract_clsvec(bert, dev_dataloader)
    #x_test, y_test = extract_clsvec(bert, test_dataloader)


    # Resets the graph
    tf.reset_default_graph()
    keras.backend.clear_session()
    
    # Defines problem
    problem = 'regression'

    # Network parameters
    predictor_train_param = dict()
    predictor_train_param['lr'] = 1e-5
    predictor_train_param['criterion'] = weighted_mse
    predictor_train_param['iterations'] = 4
    predictor_train_param['batch_size'] = 16

    dve_train_param = dict()
    dve_train_param['lr'] = 1e-3
    dve_train_param['iterations'] = 400
    dve_train_param['batch_size'] = 1000

    # Defines predictive model
    pred_model = Predictor(len(x_train[0, :]), 'reg')

    # Sets checkpoint file name
    checkpoint_file_name = './tmp/model.ckpt'

    # Flags for using stochastic gradient descent / pre-trained model
    flags = {'sgd': True, 'pretrain': False}

    # Initializes DVRL
    dvrl_class = dvrl_torch.Dvrl(x_train, y_train, x_valid, y_valid,
                        problem, pred_model, dve_train_param, predictor_train_param)

    # Trains DVRL
    dvrl_class.train_dvrl('mse')

    print('Finished dvrl training.')

    # Estimates data values
    dve_out = dvrl_class.data_valuator(x_train, y_train)

    # Predicts with DVRL
    #y_test_hat = dvrl_class.dvrl_predictor(x_test)

    print('Finished data valuation.')

if __name__ == "__main__":
    main()