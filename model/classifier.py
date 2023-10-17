import torch
import torch.nn as nn 
import numpy as np

class classifier(nn.Module):
    def __init__(self, config, events_num):
        super(classifier, self).__init__()
        self.config = config
        self.events_num = events_num
        self.embedding_dim = self.config.embedding_dim
        self.classifier = nn.Linear(self.config.hidden_dim, events_num, bias=False)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, feature, input_masks, labels):
        logits = self.classifier(feature)
        # test/dev
        if labels == None:
            return logits
        # train
        active_loss = input_masks.view(-1) == 1
        
        active_logits = logits.view(-1, self.events_num)[active_loss]
        active_labels = labels.view(-1)[active_loss]
        loss = self.criterion(active_logits, active_labels)
        
        return logits, loss
        