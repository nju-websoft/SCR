import torch
import torch.nn as nn 
from transformers import BertModel
import numpy as np
from transformers import logging
logging.set_verbosity_warning()
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torchcrf import CRF
import warnings
warnings.filterwarnings('ignore')

class entityDetection(nn.Module):

    def __init__(self, config, rnn_dim=128):
        super(entityDetection, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.2)
        self.birnn = nn.LSTM(768, rnn_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(rnn_dim*2, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)
    

    def forward(self, input_ids, labels, token_type_ids=None, input_mask=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask)
        sequence_output = outputs[0]
        sequence_output, _ = self.birnn(sequence_output)
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)
        loss = -1*self.crf(emissions, labels, mask=input_mask.byte())
        return loss

    
    def get_res(self, input_ids, token_type_ids=None, input_mask=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask)
        sequence_output = outputs[0]
        sequence_output, _ = self.birnn(sequence_output)
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)
        res = self.crf.decode(emissions, input_mask.byte())
        return res