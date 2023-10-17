import torch
import torch.nn as nn 
from transformers import BertModel
import numpy as np

class argumentDetection(nn.Module):
    def __init__(self, config):
        super(argumentDetection, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.embedding_dim = self.config.embedding_dim
        self.classifier = nn.Linear(self.embedding_dim*2, config.args_num, bias=False)
        self.dropout = nn.Dropout(0.2)
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, input_ids, labels, segment_ids, input_mask, offset, metadata, unseen_matadata, trigger, ner, gold_args):
        sequence_output = self.bert(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)[0]
        new_logits = None
        new_label = []
        for i in range(len(ner)):
            for start, end in ner[i]:
                embedding = sequence_output[i][[start+1, end]].view(-1, self.embedding_dim*2)
                embedding = self.dropout(embedding)
                logits = self.classifier(embedding)
                one_trigger = trigger[i]
                unseen_args = unseen_matadata[one_trigger]
                logits[:,unseen_args] = 0
                label = labels[i][start+1]
                new_label.append(label)
                if new_logits == None:
                    new_logits = logits
                else:
                    new_logits = torch.cat([new_logits, logits], dim = 0)

        new_label = torch.tensor(new_label).cuda()
        
        loss = self.criterion(new_logits, new_label)
        return loss

          
    def get_res(self, input_ids, segment_ids, input_mask, ner):
        sequence_output = self.bert(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)[0]
        res_logits = []
        for i in range(len(ner)):
            one_logits = None
            for start, end in ner[i]:
                embedding = sequence_output[i][[start+1, end]].view(-1, self.embedding_dim*2)
                embedding = self.dropout(embedding)
                logits = self.classifier(embedding)
                if one_logits == None:
                    one_logits = logits
                else:
                    one_logits = torch.cat([one_logits, logits], dim = 0)
            
            res_logits.append(one_logits)
        return res_logits

    def get_feature(self, input_ids, segment_ids, input_mask):
        sequence_output = self.bert(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)[0]
        feature = self.dropout(sequence_output)
        feature = feature.view((1,-1))
        return feature