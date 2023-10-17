import torch
import torch.nn as nn 
from transformers import BertModel
import numpy as np
from transformers import logging
logging.set_verbosity_warning()
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False,
                 hidden_units=None, hidden_activation='linear', hidden_initializer='xaiver', **kwargs):
        super(LayerNorm, self).__init__()
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)

        self.initialize_weights()

    def initialize_weights(self):

        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)
            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(self, inputs, cond=None):
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)
            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)  
            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs ** 2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) ** 0.5
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs

class triggerEncoder(nn.Module):
    def __init__(self, config):
        super(triggerEncoder, self).__init__()
        self.config = config
        self.last_k_attention = config.last_k_attention
        self.bert = BertModel.from_pretrained(config.bert_path, output_attentions=True)
        self.embedding_dim = self.config.embedding_dim
        self.drop = nn.Dropout(0.2)
        self.linear_transform = nn.Linear(self.bert.config.hidden_size, self.config.hidden_dim, bias=True)
        self.layer_normalization = nn.LayerNorm([self.config.hidden_dim, self.config.hidden_dim])

    def get_attention(self, input_ids, input_masks, segment_ids):
        
        output = self.bert(input_ids, token_type_ids = segment_ids, attention_mask = input_masks)
        
        now_attention = 0
        attention = output[2]
        for i in range(self.last_k_attention):
            now_layer_att = attention[-i]
            now_layer_att = torch.mean(now_layer_att, 1)
            res_att = now_layer_att/(torch.sum(now_layer_att, dim = -1, keepdim = True)+1e-9)
            now_attention += res_att
        avg_layer_att = now_attention/self.last_k_attention
        return avg_layer_att




    def get_feature(self, sentence_ids, input_ids, input_masks, segment_ids):
        feature = self.bert(input_ids, token_type_ids = segment_ids, attention_mask = input_masks)[0]
        seq_output = self.drop(feature)
        seq_output = self.linear_transform(seq_output)
        output = F.gelu(seq_output)
        feature = self.layer_normalization(output)
        feature = feature.view((1,-1))
        return feature

    def forward(self, sentence_ids, input_ids, input_masks, segment_ids):
        seq_output = self.bert(input_ids, token_type_ids = segment_ids, attention_mask = input_masks)[0]
        seq_output = self.drop(seq_output)
        seq_output = self.linear_transform(seq_output)
        output = F.gelu(seq_output)
        output = self.layer_normalization(output)
        return output
