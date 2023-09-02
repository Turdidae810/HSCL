import torch
from torch import nn
import sys
import torch.optim as optim
import numpy as np
import time
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR

from sklearn.metrics import accuracy_score, f1_score
from utils.eval_metrics import *
from utils.tools import *
from model import HSCL
import logging
from modules.loss import SupConR, Similarity, SupConLoss

import pickle

class Solver(object):  
    def __init__(self, hyp_params, train_loader, dev_loader, test_loader, is_train=True, model=None, pretrained_emb=None):
        self.hp = hp = hyp_params  # args
        self.epoch_i = 0
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader

        self.alpha = hp.alpha
        self.omega1 = hp.omega1
        self.omega2 = hp.omega2

        self.is_train = is_train
        self.model = model = HSCL(hp)

        self.update_batch = hp.update_batch  

        # initialize the model
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.criterion = criterion = nn.L1Loss(reduction="mean")  # loss 
        
        # optimizer
        self.optimizer={}

        if self.is_train:
            # mmilb_param = []
            main_param = []
            bert_param = []

            for name, p in model.named_parameters(): 
                # print(name)
                if p.requires_grad:
                    if 'bert' in name:  
                        bert_param.append(p)
                    # elif 'mi' in name:
                    #     mmilb_param.append(p)
                    else: 
                        main_param.append(p)
                
                # for p in (mmilb_param + main_param):
                for p in main_param:
                    if p.dim() > 1: # only tensor with no less than 2 dimensions are possible to calculate fan_in/fan_out
                        nn.init.xavier_normal_(p)  # xavier

        
        optimizer_main_group = [
            {'params': bert_param, 'weight_decay': hp.weight_decay_bert, 'lr': hp.lr_bert},
            {'params': main_param, 'weight_decay': hp.weight_decay_main, 'lr': hp.lr_main}
        ]

        self.optimizer_main = getattr(torch.optim, self.hp.optim)(
            optimizer_main_group
        )

        self.scheduler_main = StepLR(self.optimizer_main, step_size=11,gamma=0.1) # step_size=5 in MOSEI


    ####################################################################
    #
    # Training and evaluation scripts
    #
    ####################################################################

    def train_and_eval(self):
        model = self.model
        optimizer_main = self.optimizer_main
        scheduler_main = self.scheduler_main
        criterion = self.criterion

        def train(model, optimizer, criterion):
            epoch_loss = 0

            model.train()
            num_batches = self.hp.n_train // self.hp.batch_size
            proc_loss, proc_size = 0, 0

            start_time = time.time()
            left_batch = self.update_batch

            for i_batch, batch_data in enumerate(self.train_loader):  # 取数据
                text, visual, vlens, audio, alens, y, l, bert_sent, bert_sent_type, bert_sent_mask, ids = batch_data
                
                model.zero_grad()

                with torch.cuda.device(0):
                    text, visual, audio, y, l, bert_sent, bert_sent_type, bert_sent_mask = \
                    text.cuda(), visual.cuda(), audio.cuda(), y.cuda(), l.cuda(), bert_sent.cuda(), \
                    bert_sent_type.cuda(), bert_sent_mask.cuda()
                
                batch_size = y.size(0)
                contrast_loss=0.0

                preds,feature,fusion,feature_at,feature_vt = model(text, visual, audio, vlens, alens, bert_sent, bert_sent_type, bert_sent_mask)

                y_loss = criterion(preds, y)

                cre_cl = SupConR(temperature=0.5) 
                cre_simi = Similarity()
                Loss_fusion = cre_cl(feature,y)
                at_contrast_loss = cre_cl(feature_at,y)
                vt_contrast_loss = cre_cl(feature_vt,y)
                at_simi_loss = cre_simi(feature_at)
                vt_simi_loss = cre_simi(feature_vt)
                Loss_at = at_contrast_loss + self.alpha*at_simi_loss
                Loss_vt = vt_contrast_loss + self.alpha*vt_simi_loss
             
                loss = y_loss + self.omega1*(Loss_at + Loss_vt) + self.omega2 * Loss_fusion
                loss.backward()
                
                # -------------------------------------------------------- #
                left_batch -= 1 
                if left_batch == 0:  # self.update_batch = hp.update_batch  
                    left_batch = self.update_batch
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.hp.clip) 
                    optimizer.step()
                # -------------------------------------------------------- #

                proc_loss += loss.item() * batch_size
                proc_size += batch_size
                epoch_loss += loss.item() * batch_size

            return epoch_loss / self.hp.n_train

        def evaluate(model, criterion, test=False):
            model.eval()
            loader = self.test_loader if test else self.dev_loader
            main_loss = 0.0        
            results = []
            truths = []

            with torch.no_grad():
                for batch in loader:
                    text, vision, vlens, audio, alens, y, lengths, bert_sent, bert_sent_type, bert_sent_mask, ids = batch

                    with torch.cuda.device(0):
                        text, audio, vision, y = text.cuda(), audio.cuda(), vision.cuda(), y.cuda()
                        lengths = lengths.cuda()
                        bert_sent, bert_sent_type, bert_sent_mask = bert_sent.cuda(), bert_sent_type.cuda(), bert_sent_mask.cuda()
                
                    batch_size = lengths.size(0) # bert_sent in size (bs, seq_len, emb_size)
                    preds, features, fusion,feature_at,feature_vt = model(text, vision, audio, vlens, alens, bert_sent, bert_sent_type, bert_sent_mask)
            
                    if self.hp.dataset in ['mosi', 'mosei', 'mosei_senti'] and test:
                        criterion = nn.L1Loss()

                    main_loss += criterion(preds, y).item() * batch_size

                    # Collect the results into ntest if test else self.hp.n_valid)
                    results.append(preds)
                    truths.append(y)
    
                    # ------------------------
            
            avg_main_loss = main_loss / (self.hp.n_test if test else self.hp.n_valid)
            # avg_multi_con_loss = 0

            results = torch.cat(results)
            truths = torch.cat(truths)
        
            return avg_main_loss, results, truths

        best_valid = 1e8
        best_accuracy = 0
        best_mae = 1e8
        patience = self.hp.patience

        for epoch in range(1, self.hp.num_epochs+1):  # best_epoch=13
            start = time.time()
            logging.info(f'epoch {epoch}:')

            self.epoch = epoch

            train_main_loss = train(model, optimizer_main, criterion) 
            val_loss, results_val, truths_val = evaluate(model, criterion, test=False)           
            test_loss, results, truths = evaluate(model, criterion, test=True)       
            end = time.time()
            duration = end-start
            scheduler_main.step()
            learning_rate = optimizer_main.state_dict()['param_groups'][0]['lr']

            logging.info("-"*50)
            logging.info('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, val_loss, test_loss))
            logging.info("-"*50)
            print("-"*50)
            print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, val_loss, test_loss))
            print("-"*50)
            # --------------日志结束------------------
            if val_loss < best_valid:
                # update best validation
                patience = self.hp.patience
                best_valid = val_loss
                if test_loss < best_mae:
                    best_epoch = epoch
                    best_mae = test_loss
                    if self.hp.dataset in ["mosei_senti", "mosei"]:
                        eval_mosei_senti(results, truths, True)
                    elif self.hp.dataset == 'mosi':
                        eval_mosi(results, truths, True)
                    best_results = results
                    best_truths = truths
                    print(f"Saved model at pre_trained_models/MM.pt!")
                    save_model(self.hp, model)

            else:
                patience -= 1
                if patience == 0:
                    break

        logging.info(f'Best epoch: {best_epoch}')

        if self.hp.dataset in ["mosei_senti", "mosei"]:
            eval_mosei_senti(best_results, best_truths, True)
        elif self.hp.dataset == 'mosi':
            self.best_dict = eval_mosi(best_results, best_truths, True)
     
        sys.stdout.flush()  
