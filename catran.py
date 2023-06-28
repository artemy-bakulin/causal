import torch
from torch import nn
import torch.functional as F
from torch.utils.data import random_split, DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import lightning.pytorch as pl
from lion_pytorch import Lion

import pandas as pd
import numpy as np



def init_weights(m, activation='relu'):
    if isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    
    if activation == 'silu':
        if isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2 / n))
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0) 
        elif isinstance(m, nn.Embedding):
            m.weight.data.normal_(0, 0.001)
          
    else:
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.01) 
        elif isinstance(m, nn.Embedding):
            torch.nn.init.kaiming_uniform_(m.weight)
            

class MaskedAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.ll = nn.Linear(input_dim, input_dim)
        self.ll2 = nn.Linear(input_dim, input_dim)
        
        
        
    def _choose_function(self, function):
        if function == 'sigmoid':
            func = torch.sigmoid
        elif function == 'tanh':
            func = torch.tanh
        elif function == 'softmax':
            func = lambda x: torch.softmax(x, -1)
        elif function == 'softmax_t':
            func = lambda x: torch.softmax(x, -2)

        return func
                    
    def forward(self, x, y, hidden_idxs=None, function='sigmoid'):
        x = self.ll(x)       
        
        
        if x.dim() == 3:
            att = torch.einsum('btk, blk -> btl', x, x)  
        elif x.dim() == 2:
            att = torch.einsum('tk, lk -> tl', x, x)            
            
        func = self._choose_function(function)
        
        att = func(att)
                
        return att 
    
    
class BatchNormRotating(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.embed_size = embed_size
        self.bn = nn.BatchNorm1d(self.embed_size)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        return x


class AttBlock(nn.Module):
    def __init__(self, input_size, embed_size):
        super().__init__()
    
        self.input_size = input_size
        self.embed_size = embed_size
        
        self.act = nn.SiLU()
        self.ll1 = nn.Linear(self.embed_size, self.embed_size)

        
        self.bn_attn1 = BatchNormRotating(self.embed_size)
        self.bn1 = BatchNormRotating(self.embed_size)
        
    def forward(self, x, attn_weights):
    
        attn_output1 = torch.einsum('bjk, lj -> blk', x, attn_weights)
        
        
        #attn_output1 = (attn_output1 + x)  / 2 ** 0.5
        attn_output1 = self.bn_attn1(attn_output1)

        ll_output1 = self.ll1(attn_output1)
        #ll_output1 = (attn_output1 + ll_output1)  / 2 ** 0.5
        ll_output1 = self.bn1(ll_output1)
        ll_output1 = self.act(ll_output1)
        
        return ll_output1
    
    
class Catran(pl.LightningModule):
    def __init__(self, genes, n_genes_in_minibatch, 
                     embed_size=20, lr=0.001, perc_hidden=0.25, wd=0.01, regime='interventional'):
        super().__init__()
        self.genes = genes
        self.n_genes = len(self.genes)
        self.n_genes_in_minibatch = n_genes_in_minibatch
        self.embed_size = embed_size
        self.lr = lr
        self.wd = wd
        self.perc_hidden = perc_hidden
        self.calculate_attention = True
        self.regime = regime
        
        self.act = nn.SiLU()
        self.embedding = nn.Embedding(self.n_genes, embed_size)

        self.ll0 = nn.Linear(self.embed_size+1, self.embed_size+1)
        self.bn0 = BatchNormRotating(self.embed_size+1)
        
        self.ll_pre = nn.Linear(self.embed_size+1, self.embed_size+1)
        self.bn_pre = BatchNormRotating(self.embed_size+1)
        
        
        self.attn = MaskedAttention(self.embed_size)
       
        self.att_block1 = AttBlock(self.n_genes_in_minibatch, self.embed_size+1)


        self.ll_prefin = nn.Linear(self.embed_size+1, self.embed_size+1)
        self.bn_prefin = BatchNormRotating(self.embed_size+1)
        

        self.ll_fin = nn.Linear(self.embed_size+1, 1)
        
        self.model = nn.ModuleList(
                             [self.embedding, self.attn, 
                              self.ll0, self.bn0,
                              self.ll_pre, self.bn_pre,
                              self.att_block1,                           
                              self.ll_prefin, self.bn_prefin, self.ll_fin]
                    )
        
        self.loss_fn = nn.HuberLoss()
        self.save_hyperparameters(logger=False)
        self.model.apply(lambda x: init_weights(x, activation='silu'))
                
        
    def _common_step(self, batch, batch_idx):
        x = batch.float()
        if self.regime == 'interventional':
            interventions = x[:, -1].int()
            x = x[:, :-1]
                
        
        gene_mask = self._generate_random_mask(self.n_genes, self.n_genes_in_minibatch)
        gene_idxs = torch.where(gene_mask)[0]
        x = x[:, gene_mask]

        self.hidden_mask = self._generate_2D_random_mask(self.n_genes_in_minibatch, 
                                     int(self.n_genes_in_minibatch * self.perc_hidden), len(x))
        
            
        x_corrupted = torch.clone(x)
        x_corrupted[self.hidden_mask] = x_corrupted[self.hidden_mask][torch.randperm(len(x_corrupted[self.hidden_mask]))]
        
        preds, attn_weights = self.forward(x_corrupted, gene_idxs)
        
        
            
        loss =  .7 * self.loss_fn(preds[self.hidden_mask], x[self.hidden_mask]) + \
                    .3 * self.loss_fn(preds[~self.hidden_mask], x[~self.hidden_mask])
        
        if self.regime == 'interventional':
            loss += self._calculate_interventional_loss(x, preds, self.hidden_mask, attn_weights, interventions, gene_idxs)
  
        return loss, preds[self.hidden_mask], x[self.hidden_mask]
    
    
    def _calculate_interventional_loss(self, x, preds, hidden_mask, attention_matrix, interventions, gene_idxs):
        
        huber = nn.HuberLoss(reduction='none')
        mse = nn.MSELoss(reduction='none')
        sigmoid = nn.Sigmoid()
        softmax = nn.Softmax()
        tanh = nn.Tanh()
        
        intervention_types = torch.unique(interventions)
        
        l = 0

        for interven in intervention_types:
            if interven in gene_idxs:
                interv_x = x[interventions==interven]
                non_interv_x = x[interventions!=interven]
                interv_preds = preds[interventions==interven]
                non_interv_preds = preds[interventions!=interven]
                
                
                m1 = huber(interv_x, interv_preds).mean(0)
                m2 = huber(non_interv_x, non_interv_preds).mean(0)

                
                coefs = sigmoid(m2 / m1 - 1)
                
                l += mse(attention_matrix[:, gene_idxs==interven], coefs).mean()
                
        l = l / (len(intervention_types) - 1)

        return l
    
    
        
    def forward(self, x, gene_idxs):

        x_emb = self.embedding(gene_idxs)


        attn_weights = self.attn(x_emb, x_emb, function='softmax')

        x_emb = x_emb[None, :, :].repeat((x.shape[0], 1, 1))
        x = x[:, :, None]
        x = torch.concat((x, x_emb), 2)
        
        x = self.ll0(x)
        x = self.bn0(x)
                            
        x = self.ll_pre(x)
        x = self.bn_pre(x)
        
        attn_output1 = self.att_block1(x, attn_weights)

        combined = (attn_output1) 
        
        combined = self.ll_prefin(combined)

        
        res = self.ll_fin(combined)
        res = res.squeeze(2)
        
        return res, attn_weights
    
    
    def _generate_random_mask(self, total_size, sample_size):
        mask = torch.full((total_size,), False, dtype=bool)
        mask[:sample_size] = True
        mask = mask[torch.randperm(total_size)].to(self.device)
        return mask
    
    def _generate_2D_random_mask(self, total_size, sample_size, bs):
        mask = torch.full((bs, total_size), False, dtype=bool)
        
        col_idxs = torch.concat([torch.randperm(total_size)[:sample_size] for i in range(bs)])
        row_idxs = torch.tile(torch.arange(bs), (sample_size,))
        
        mask[row_idxs, col_idxs] = True
        
        mask = mask.to(self.device)
        
        return mask
    
    def calculate_attention_matrix(self):
        x_emb = self.embedding(torch.arange(self.n_genes).to(self.device))
        self.attention_matrix = self.attn(x_emb, x_emb, function='softmax')
        self.attention_matrix = self.attention_matrix.detach().cpu().numpy()
        
    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss
        
    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log('test_loss', loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        return scores, y
    
    def configure_optimizers(self, optimizer='lion', use_scheduler=True):
        if optimizer=='adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        elif optimizer=='lion':
            optimizer = Lion(self.parameters(), lr=self.lr, weight_decay=self.wd)
        if use_scheduler:
            scheduler = MultiStepLR(optimizer, milestones=[100], gamma=0.01)
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
        else:
            return optimizer
    
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()
        
    
    
