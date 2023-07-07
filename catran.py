from abc import abstractmethod
from typing import List, Tuple

import numpy as np
from causalscbench.models.abstract_model import AbstractInferenceModel
from causalscbench.models.training_regimes import TrainingRegime

import lightning.pytorch as pl
import networkx as nx


import torch
from torch import nn
import torch.functional as F
from torch.utils.data import random_split, DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from lion_pytorch import Lion
from pytorch_metric_learning.samplers import  MPerClassSampler

import pandas as pd





class Dataset(pl.LightningDataModule):
    def __init__(self, expression_matrix, gene_names, interventions=None, regime='interventional', batch_size=2048, num_workers=1, train_split=0.9):

        super().__init__()
                
        self.expression_matrix = expression_matrix
        self.batch_size = batch_size
        self.train_split = train_split
        self.num_workers = num_workers
        
        
        
        self.genes = np.array(gene_names)
        self.genes, expr_mask = np.unique(gene_names, return_index=True)
        
        
        self.expression_matrix = expression_matrix[:, expr_mask]

        self.n_genes = len(self.genes)
        self.n_cells = self.expression_matrix.shape[0]
        
        
        self.interventions = np.array(interventions)
        self.regime = regime
            
            
        if self.regime == 'interventional':
            
            
            get_gene_idx = lambda x: np.where(self.genes == x)[0][0] if x in self.genes else -1

            pert_idxs = pd.Series(self.interventions).apply(get_gene_idx).values
            
            self.expression_matrix = np.concatenate((self.expression_matrix, pert_idxs[:, None]), axis=1)
            
            
            
    def setup(self, stage=None):    
        self.train = self.expression_matrix
        

    def train_dataloader(self):
        sampler = MPerClassSampler((self.train[:, -1]), 25)

        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          sampler=sampler, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)
    



class MyModel(AbstractInferenceModel):
    
    def __init__(self,
                batch_size=2048,
                N_SAMPLE_GENES=500,
                LR = .0001,
                WD = 0.05,
                PERC_HIDDEN = 0.80,
                NUM_EPOCHS = 50,
                top_n_edges = 1_000,
                accelerator='gpu'):
        super().__init__()
        
        self.batch_size=batch_size
        self.N_SAMPLE_GENES=N_SAMPLE_GENES
        self.LR = LR
        self.WD=WD
        self.PERC_HIDDEN = PERC_HIDDEN
        self.NUM_EPOCHS = NUM_EPOCHS
        self.top_n_edges = top_n_edges
        self.accelerator = accelerator
        
    
    
    def __call__(
        self,
        expression_matrix: np.array,
        interventions: List[str],
        gene_names: List[str],
        training_regime: TrainingRegime,
        seed: int = 0,
    ) -> List[Tuple]:
                

        data = Dataset(expression_matrix, gene_names, batch_size=self.batch_size, interventions=interventions)
        
        
        N_SAMPLE_GENES = min(self.N_SAMPLE_GENES, len(data.genes))
        
        model = Catran(data.genes, N_SAMPLE_GENES, lr=self.LR, perc_hidden=self.PERC_HIDDEN, wd=self.WD)
        
        trainer = pl.Trainer(min_epochs=1, max_epochs=self.NUM_EPOCHS, accelerator=self.accelerator, limit_val_batches=0)
        
        trainer.fit(model, data)
        
        
        model.calculate_attention_matrix()
        att = pd.DataFrame(model.attention_matrix, index=data.genes, columns=data.genes)
        att.iloc[range(len(att)), range(len(att))] = 0
                
        att = np.maximum(att, att.T)
        att.iloc[:, :] = np.triu(att)
        
        nth_el = np.sort(att.values.flatten())[-self.top_n_edges]

        Adj = (att >= nth_el).astype(int)
                
            
        
        edges = []
        with open('edges.txt', 'w+') as o:
            for gene_A in Adj.index:
                for gene_B in Adj.columns:
                    if Adj.loc[gene_A, gene_B]:
                        edges.append((gene_A, gene_B))
                        o.write(gene_A + '_' + gene_B + '\n')
                    
                            
        return edges
    
    
    
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
            m.weight.data.normal_(0, 0.0001)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0) 
        elif isinstance(m, nn.Embedding):
            m.weight.data.normal_(0, 0.1)
          
    else:
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.01) 
        elif isinstance(m, nn.Embedding):
            torch.nn.init.kaiming_uniform_(m.weight)
            

class MaskedAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        #self.ll = nn.Linear(input_dim, input_dim)
        #self.act = nn.SiLU()
        
        
        
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
                    
    def forward(self, x, y, hidden_idxs=None, function='softmax'):
        
        #x = self.ll(x)
        #x = self.act(x)
        
        
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

        #ll_output1 = self.ll1(attn_output1)
        #ll_output1 = (attn_output1 + ll_output1)  / 2 ** 0.5
        #ll_output1 = self.bn1(ll_output1)
        #ll_output1 = self.act(ll_output1)
        
        return attn_output1
    
    
class Catran(pl.LightningModule):
    def __init__(self, genes, n_genes_in_minibatch, 
                embed_size=40, lr=0.001, perc_hidden=0.25, wd=0.01, use_2nd_att=False,
                kappa=0.4, phi=1, interventional=True):
        super().__init__()
        self.genes = genes
        self.n_genes = len(self.genes)
        self.n_genes_in_minibatch = n_genes_in_minibatch
        self.embed_size = embed_size
        self.lr = lr
        self.wd = wd
        self.perc_hidden = perc_hidden
        self.calculate_attention = True
        self.kappa = kappa
        self.phi = phi
        self.interventional = interventional
        
        self.use_2nd_att = use_2nd_att
        
        self.act = nn.SiLU()
        self.embedding = nn.Embedding(self.n_genes, embed_size)

        self.ll0 = nn.Linear(self.embed_size+1, self.embed_size+1)
        self.bn0 = BatchNormRotating(self.embed_size+1)
        
        self.ll_pre = nn.Linear(self.embed_size+1, self.embed_size+1)
        self.bn_pre = BatchNormRotating(self.embed_size+1)
        
        
        self.attn = MaskedAttention(self.embed_size)
       
        self.att_block1 = AttBlock(self.n_genes_in_minibatch, self.embed_size+1)
        self.att_block2 = AttBlock(self.n_genes_in_minibatch, self.embed_size+1)


        self.ll_prefin = nn.Linear(self.embed_size+1, self.embed_size+1)
        self.bn_prefin = BatchNormRotating(self.embed_size+1)
        

        self.ll_fin = nn.Linear(self.embed_size+1, 1)
        
        self.model = nn.ModuleList(
                             [self.embedding, self.attn, 
                              self.ll0, self.bn0,
                              self.ll_pre, self.bn_pre,
                              self.att_block1, self.att_block2,                           
                              self.ll_prefin, self.bn_prefin, self.ll_fin]
                    )
        
        self.loss_fn = nn.HuberLoss()
        self.save_hyperparameters(logger=False)
        self.model.apply(lambda x: init_weights(x, activation='silu'))
                
        
    def _common_step(self, batch, batch_idx):
        x = batch.float().to('cpu')
        
        if self.interventional:
            interventions = x[:, -1].int().to(self.device)
            x = x[:, :-1]
                        
        gene_mask = self._generate_random_mask(self.n_genes, self.n_genes_in_minibatch)
        gene_idxs = torch.where(gene_mask)[0].to(self.device)
        x = x[:, gene_mask].to(self.device)

        self.hidden_mask = self._generate_2D_random_mask(self.n_genes_in_minibatch, 
                                     int(self.n_genes_in_minibatch * self.perc_hidden), len(x)).to(self.device)
        
            
        x_corrupted = torch.clone(x)
        x_corrupted[self.hidden_mask] = x_corrupted[self.hidden_mask][torch.randperm(len(x_corrupted[self.hidden_mask]))]
        
        preds, attn_weights = self.forward(x_corrupted, gene_idxs)
        
        
            
        loss =  self.loss_fn(preds[self.hidden_mask], x[self.hidden_mask]) + \
                    self.kappa * self.loss_fn(preds[~self.hidden_mask], x[~self.hidden_mask])
        
        if self.interventional:
            loss += 10 * self._calculate_interventional_loss(x, preds, self.hidden_mask, attn_weights, interventions, gene_idxs)
  
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
                
                l += mse(attention_matrix[:, gene_idxs==interven].squeeze(1), coefs).mean()
                
                
                
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
        
#         x = self.act(x)
                            
#         x = self.ll_pre(x)
#         x = self.bn_pre(x)
        
        attn_output1 = self.att_block1(x, attn_weights)
        combined = attn_output1
        for i in range(10):
            combined =  self.att_block1(combined, attn_weights)
        #combined = self.att_block1(combined, attn_weights)
        #combined = self.att_block1(combined, attn_weights)
        
        if self.use_2nd_att:
            attn_output2 = self.att_block1(x, attn_weights)
            combined += attn_output2
        
        combined = self.ll_prefin(combined)
        combined = self.bn_prefin(combined)
        combined = self.act(combined)
        
        res = self.ll_fin(combined)
        res = res.squeeze(2)
        
        return res, attn_weights
    
    
    def _generate_random_mask(self, total_size, sample_size):
        mask = torch.full((total_size,), False, dtype=bool)
        mask[:sample_size] = True
        mask = mask[torch.randperm(total_size)]
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
    
    def configure_optimizers(self, optimizer='lion', use_scheduler=False):
        if optimizer=='adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        elif optimizer=='lion':
            optimizer = Lion(self.parameters(), lr=self.lr, weight_decay=self.wd)
        if use_scheduler:
            scheduler = MultiStepLR(optimizer, milestones=[100], gamma=0.01)
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
        else:
            return [optimizer]
    
#     def lr_scheduler_step(self, scheduler, metric):
#         scheduler.step()
        