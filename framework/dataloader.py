import torch
from torch.utils.data import Dataset, DataLoader
import os
import copy
import numpy as np
import random
import json
from framework.utils import trigger_combine_event, args_combine_event
from transformers import BertTokenizer
from transformers import logging
logging.set_verbosity_warning()
logging.set_verbosity_error()

class ACETriDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def collate_fn(self, data):
        sentence_ids = [torch.tensor(item['sentence_ids']) for item in data]
        input_ids = [torch.tensor(item['input_ids']) for item in data]
        input_masks = [torch.tensor(item['input_masks']) for item in data]
        in_sent = [torch.tensor(item['in_sent']) for item in data]
        segment_ids = [torch.tensor(item['segment_ids']) for item in data]
        labels = [torch.tensor(item['labels']) for item in data]
        ners = [torch.tensor(item['ners']) for item in data]
        sentence = [item['sentence'] for item in data]
        return (sentence_ids, input_ids, input_masks, in_sent, segment_ids, labels, ners, sentence)

def get_ACETriData_loader(data, config, shuffle = False, batch_size = None):
    dataset = ACETriDataset(data)
    if batch_size == None:
        batchSize = min(config.batch_size, len(data))
    else:
        batchSize = min(batch_size, len(data))
    ACETriData_loader = DataLoader(
        dataset = dataset,
        batch_size= batchSize,
        shuffle= shuffle,
        collate_fn= dataset.collate_fn,
        drop_last= False
        )
    return ACETriData_loader

class ACEPredDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def collate_fn(self, data):
        input_ids = [torch.tensor(item['input_ids']) for item in data]
        input_masks = [torch.tensor(item['input_masks']) for item in data]
        in_sent = [torch.tensor(item['in_sent']) for item in data]
        segment_ids = [torch.tensor(item['segment_ids']) for item in data]
        sentence = [item['sentence'] for item in data]
        event = [item['event'] for item in data]
        return (input_ids, input_masks, in_sent, segment_ids, sentence, event)

def get_ACEPredData_loader(data, config, shuffle = False, batch_size = None):
    dataset = ACEPredDataset(data)
    if batch_size == None:
        batchSize = min(config.batch_size, len(data))
    else:
        batchSize = min(batch_size, len(data))
    ACEPredData_loader = DataLoader(
        dataset = dataset,
        batch_size= batchSize,
        shuffle= shuffle,
        collate_fn= dataset.collate_fn,
        drop_last= False
        )
    return ACEPredData_loader

def get_ACEArgData_loader(data, config, shuffle = False, batch_size = None):
    dataset = ACEArgDataloader(data)
    if batch_size == None:
        batchSize = min(config.batch_size, len(data))
    else:
        batchSize = min(batch_size, len(data))
    ACEPredData_loader = DataLoader(
        dataset = dataset,
        batch_size= batchSize,
        shuffle= shuffle,
        collate_fn= dataset.collate_fn,
        drop_last= False
        )
    return ACEPredData_loader

class ACEArgDataloader(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def collate_fn(self, data):

        sentence = [item['sentence'] for item in data]
        input_ids = [torch.tensor(item['input_ids']) for item in data]
        input_masks = [torch.tensor(item['input_masks']) for item in data]
        in_sent = [torch.tensor(item['in_sent']) for item in data]
        segment_ids = [torch.tensor(item['segment_ids']) for item in data]
        args =  [torch.tensor(item['args']) for item in data]
        args_offset = [item['args_offset'] for item in data]
        gold_args = [item['gold_args'] for item in data]
        ner = [item['ner'] for item in data]
        trigger = [item['trigger'] for item in data]

        

        return (sentence, input_ids, input_masks, in_sent, segment_ids, args, args_offset, gold_args, ner, trigger)

class ACEPredNerDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def collate_fn(self, data):

        
        input_ids = [torch.tensor(item['input_ids']) for item in data]
        input_masks = [torch.tensor(item['input_masks']) for item in data]
        in_sent = [torch.tensor(item['in_sent']) for item in data]
        segment_ids = [torch.tensor(item['segment_ids']) for item in data]
        sentence = [item['sentence'] for item in data]
        event = [item['event'] for item in data]
        ner = [item['ner'] for item in data]


        return (input_ids, input_masks, in_sent, segment_ids, sentence, event, ner)

def get_ACEPredNerData_loader(data, config, shuffle = False, batch_size = None):
    dataset = ACEPredNerDataset(data)
    if batch_size == None:
        batchSize = min(config.batch_size, len(data))
    else:
        batchSize = min(batch_size, len(data))
    ACEPredNerData_loader = DataLoader(
        dataset = dataset,
        batch_size= batchSize,
        shuffle= shuffle,
        collate_fn= dataset.collate_fn,
        drop_last= False
        )
    return ACEPredNerData_loader

class ACETriDataloder(Dataset):
    def __init__(self, config, i):
        
        self.config = config
        self.data_root = config.data_root
        #print(config.bert_path, type(config.bert_path))
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_path)
        self.max_sentence_length = 512

        # trigger category vocabulary
        self.vocab2index = {}
        self.index2vocab = {}
        self.vocab2index = json.load(open(self.data_root+'label2id.json', 'r'))
        self.vocab2index['None'] = 0
        for key, value in self.vocab2index.items():
            #value = value - 169
            self.index2vocab[value] = key
            #self.vocab2index[key] = value

        # ner2id
        self.ner2id = json.load(open(self.data_root+'ner2id.json', 'r'))
        self.ner2id['None'] = 0
        self.id2ner = {}
        for key, value in self.ner2id.items():
            self.id2ner[value] = key
        
        # iter
        self.stream_turn = config.stream_turn
        self.batch = 0

        # data stream
        self.train_stream = json.load(open(self.data_root+'train_streams.json', 'r'))
        self.id2stream = json.load(open(self.data_root+'id2stream.json', 'r'))

       

        # set seed
        self.seed = config.seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.shuffle_index = list(range(self.stream_turn))
        random.shuffle(self.shuffle_index)
        self.shuffle_index = np.argsort(self.shuffle_index)
        
        #self.shuffle_index = PERM[i]
        print(self.shuffle_index)

        # seen data
        self.seen_test_data = []
        self.seen_dev_data = []
        self.seen_event = []

        self.seen_test_args_data = []
        self.seen_dev_args_data = []
        self.seen_args = []

        self.ltlabel = []

        # prepare data:
        self.train_dataset, self.dev_dataset, self.test_dataset = self.read_data(self.data_root)
        
        # tokenizer:
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_path)

        if config.argument:

            # role2id
            self.role2id = json.load(open(self.data_root+'role2id.json', 'r'))
            self.role2id['None'] = 0
            self.id2role = {}
            for key, value in self.role2id.items():
                self.id2role[value] = key
            
             # metadata
            self.args_num = config.args_num
            self.metadata = json.load(open(self.data_root+'metadata.json', 'r'))
            self.unseen_metadata = {}
            for key, value in self.metadata.items():
                new_value = [self.role2id[val] for val in value]
                self.metadata[key] = new_value
                unseen_args = [i for i in range(self.args_num)]
                unseen_args = list(set(unseen_args) - set(new_value) - set([0]))
                self.unseen_metadata[key] = unseen_args
            self.args_train_dataset, self.args_dev_dataset, self.args_test_dataset = self.read_args_data(self.data_root)

        if config.lttest:
            self.ltlabel = json.load(open(self.data_root+'lt_label.json', 'r'))

    def __iter__(self):
        return self
    
    def __len__(self):
        return self.stream_turn

    def __next__(self):
        cur_train_data, cur_test_data, cur_dev_data, current_event = [], [], [], []
        if self.batch == self.stream_turn:
            self.batch = 0
            raise StopIteration()
        index = self.shuffle_index[self.batch]

        # now is tirgger data

        cur_train_data = self.train_dataset[index]
        cur_dev_data = self.dev_dataset[index]
        cur_test_data = self.test_dataset[index]
        current_event = self.train_stream[index]
        self.seen_event += current_event
        
        self.batch += 1
        
        final_data = [[] , [] , []]
        for i, data in enumerate([cur_train_data, cur_dev_data, cur_test_data]):
            for x in data:
                final_data[i] += data[x]
        tr_data, de_data, cur_te_data = final_data
        tr_data = trigger_combine_event([], tr_data)
        de_data = trigger_combine_event([], de_data)
        cur_te_data = trigger_combine_event([], cur_te_data)
        temp_cur = copy.deepcopy(cur_te_data)
        temp_dev = copy.deepcopy(de_data)
        self.seen_test_data = trigger_combine_event(self.seen_test_data, temp_cur)
        self.seen_dev_data = trigger_combine_event(self.seen_dev_data, temp_dev)


        if self.config.argument:
            # now is args data
            cur_args_train_data = self.args_train_dataset[index]
            cur_args_dev_data = self.args_dev_dataset[index]
            cur_args_test_data = self.args_test_dataset[index]
            #current_args = self.args_stream[index]
            #self.seen_args = list(set(self.seen_args + current_args))
            #unseen_args = [i for i in range(self.args_num)]
            #unseen_args = list(set(unseen_args) - set(self.seen_args)- set([0]))
            args_final_data = [[] , [] , []]
            for i, data in enumerate([cur_args_train_data, cur_args_dev_data, cur_args_test_data]):
                for x in data:
                    args_final_data[i] += data[x]
            tr_args_data, de_args_data, cur_args_te_data = args_final_data
            tr_args_data = args_combine_event([], tr_args_data)
            de_args_data = args_combine_event([], de_args_data)
            cur_args_te_data = args_combine_event([], cur_args_te_data)
            temp_cur = copy.deepcopy(cur_args_te_data)
            temp_dev = copy.deepcopy(de_args_data)
            self.seen_test_args_data = args_combine_event(self.seen_test_args_data, temp_cur)
            self.seen_dev_args_data = args_combine_event(self.seen_dev_args_data, temp_dev)

            return tr_data, de_data, cur_te_data, current_event, self.seen_event, self.seen_test_data, cur_train_data, self.seen_dev_data, \
                tr_args_data, de_args_data, cur_args_te_data, self.seen_test_args_data, self.seen_dev_args_data, cur_args_train_data
        
        return tr_data, de_data, cur_te_data, current_event, self.seen_event, self.seen_test_data, cur_train_data, self.seen_dev_data, \
                None, None, None, None, None, None

    def read_data(self, data_root):
        train_data = json.load(open(data_root+'train.json', 'r'))
        dev_data = json.load(open(data_root+'dev.json', 'r'))
        test_data = json.load(open(data_root+'test.json', 'r'))

        train_dataset = [{} for i in range(self.stream_turn)]
        dev_dataset = [{} for i in range(self.stream_turn)]
        test_dataset = [{} for i in range(self.stream_turn)]

        train_dataset = self.load_dataset(train_data, train_dataset, 'train')
        dev_dataset = self.load_dataset(dev_data, dev_dataset, 'dev')
        test_dataset = self.load_dataset(test_data, test_dataset, 'test')
        
        return train_dataset, dev_dataset, test_dataset

    def load_dataset(self, train_data, train_dataset, type):
        index = 0
        for _ , event_type in enumerate(train_data.keys()):
            use_sample = train_data[event_type]
            i = int(self.id2stream[str(self.vocab2index[event_type])])
            train_dataset[i][event_type] = self.read_sample(index, use_sample, type)
            index += len(use_sample)
        return train_dataset

    def read_sample(self, index, sample, type):
        res = []
        sentence_ids = 0
        for i, asample in enumerate(sample):
            one_sample = {}
            if type == 'train':
                sentence_ids = i + index
            elif type == 'dev':
                sentence_ids = i + index + 4000
            elif type == 'test':
                sentence_ids = i + index + 6000
            sentence, event, s_start, ner = asample['sentence'], asample['event'][0], asample['s_start'], asample['ner']
            
            offset2trigger = {}
            start_offset, end_offset, trigger = event['start'] - s_start, event['end'] - s_start, event['type']
            for i in range(start_offset, end_offset):
                offset2trigger[i] = trigger
            
            offset2ner = {}
            for ne in ner:
                ner_start_offset, ner_end_offset, ner_type = ne['start'] - s_start, ne['end'] - s_start, ne['entity_type']
                for i in range(ner_start_offset, ner_end_offset):
                    offset2ner[i] = ner_type
            tokens, labels, in_sent, segment_ids, ners = [], [], [], [], []
            # sentence
            tokens.append('[CLS]')
            labels.append(self.vocab2index['None'])
            in_sent.append(0)
            segment_ids.append(0)
            ners.append(self.ner2id['None'])
            for (i, token) in enumerate(sentence):
                atoken = self.tokenizer.tokenize(token)
                if len(atoken) == 0:
                    tokens.append('[UNK]')
                else:
                    tokens.append(atoken[0])
                in_sent.append(1)
                segment_ids.append(1)
                if i in offset2trigger:
                    labels.append(self.vocab2index[offset2trigger[i]])
                else:
                    labels.append(self.vocab2index['None'])
                if i in offset2ner:
                    ners.append(self.ner2id[offset2ner[i]])
                else:
                    ners.append(self.ner2id['None'])
            tokens.append('[SEP]')
            labels.append(self.vocab2index['None'])
            in_sent.append(0)
            segment_ids.append(1)
            ners.append(self.ner2id['None'])
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_masks = [1] * len(input_ids)
            while len(input_ids) < self.max_sentence_length:
                input_ids.append(0)
                input_masks.append(0)
                in_sent.append(0)
                segment_ids.append(0)
                labels.append(self.vocab2index['None'])
                ners.append(self.ner2id['None'])
            #labels = np.array(labels) - 169
            one_sample['sentence_ids'] = sentence_ids
            one_sample['input_ids'] = input_ids
            one_sample['input_masks'] = input_masks
            one_sample['labels'] = labels
            one_sample['in_sent'] = in_sent
            one_sample['segment_ids'] = segment_ids
            one_sample['ners'] = ners
            one_sample['sentence'] = sentence
            res.append(one_sample)
        return res

    def read_pred_sample(self, file):
        data = json.load(open(file, 'r'))
        res = []
        for i, asample in enumerate(data):
            one_sample = {}
            sentence, event, s_start = asample['sentence'], asample['trigger'], asample['s_start']
            
            
            tokens, in_sent, segment_ids = [], [], []
            # sentence
            tokens.append('[CLS]')
            in_sent.append(0)
            segment_ids.append(0)
            for (i, token) in enumerate(sentence):
                atoken = self.tokenizer.tokenize(token)
                if len(atoken) == 0:
                    tokens.append('[UNK]')
                else:
                    tokens.append(atoken[0])
                in_sent.append(1)
                segment_ids.append(1)
            tokens.append('[SEP]')
            in_sent.append(0)
            segment_ids.append(1)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_masks = [1] * len(input_ids)
            while len(input_ids) < self.max_sentence_length:
                input_ids.append(0)
                input_masks.append(0)
                in_sent.append(0)
                segment_ids.append(0)
            #labels = np.array(labels) - 169
            one_sample['input_ids'] = input_ids
            one_sample['input_masks'] = input_masks
            one_sample['in_sent'] = in_sent
            one_sample['segment_ids'] = segment_ids
            one_sample['sentence'] = sentence
            one_sample['event'] = event
            res.append(one_sample)
        return res

    def read_pred_ner_sample(self, file):
        data = json.load(open(file, 'r'))
        res = []
        for i, asample in enumerate(data):
            one_sample = {}
            sentence, event, s_start, ner = asample['sentence'], asample['trigger'], asample['s_start'], asample['ner']
            
            
            tokens, in_sent, segment_ids = [], [], []
            # sentence
            tokens.append('[CLS]')
            in_sent.append(0)
            segment_ids.append(0)
            for (i, token) in enumerate(sentence):
                atoken = self.tokenizer.tokenize(token)
                if len(atoken) == 0:
                    tokens.append('[UNK]')
                else:
                    tokens.append(atoken[0])
                in_sent.append(1)
                segment_ids.append(1)
            tokens.append('[SEP]')
            in_sent.append(0)
            segment_ids.append(1)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_masks = [1] * len(input_ids)
            while len(input_ids) < self.max_sentence_length:
                input_ids.append(0)
                input_masks.append(0)
                in_sent.append(0)
                segment_ids.append(0)
            #labels = np.array(labels) - 169
            one_sample['input_ids'] = input_ids
            one_sample['input_masks'] = input_masks
            one_sample['in_sent'] = in_sent
            one_sample['segment_ids'] = segment_ids
            one_sample['sentence'] = sentence
            one_sample['event'] = event
            one_sample['ner'] = ner
            res.append(one_sample)
        return res

    def read_args_data(self, data_root):
        train_data = json.load(open(data_root+'train.json', 'r'))
        dev_data = json.load(open(data_root+'dev.json', 'r'))
        test_data = json.load(open(data_root+'test.json', 'r'))

        train_dataset = [{} for i in range(self.stream_turn)]
        dev_dataset = [{} for i in range(self.stream_turn)]
        test_dataset = [{} for i in range(self.stream_turn)]

        train_dataset = self.load_args_dataset(train_data, train_dataset, 'train')
        dev_dataset = self.load_args_dataset(dev_data, dev_dataset, 'dev')
        test_dataset = self.load_args_dataset(test_data, test_dataset, 'test')

        return train_dataset, dev_dataset, test_dataset

    def load_args_dataset(self, train_data, train_dataset, type):
        index = 0
        for _ , event_type in enumerate(train_data.keys()):
            use_sample = train_data[event_type]
            i = int(self.id2stream[str(self.vocab2index[event_type])])
            train_dataset[i][event_type] = self.read_args_sample(index, use_sample, type)
            index += len(use_sample)
        return train_dataset

    def read_args_sample(self, index, sample, type):
        res = []
        for i, asample in enumerate(sample):
            one_sample = {}
            sentence, event, s_start, args, ners = asample['sentence'], asample['event'][0]['type'], asample['s_start'], asample['event'][1], asample['ner']
            args_offset = []
            gold_args = []
            offset2args = {}
            ner = []
            for index, ne in enumerate(ners):
                start, end = ne['start'], ne['end']
                for i in range(start, end):
                    args_offset.append(i)
                ner.append([start, end])
            for ar in args:
                args_start_offset, args_end_offset, args_type = ar['start'] - s_start, ar['end'] - s_start, ar['role']
                gold_args.append((args_start_offset, args_end_offset, self.role2id[args_type]))
                for i in range(args_start_offset, args_end_offset):
                    offset2args[i] = args_type
            tokens, in_sent, segment_ids, args = [], [], [], []
            # sentence
            tokens.append('[CLS]')
            args.append(self.role2id['None'])
            in_sent.append(0)
            segment_ids.append(0)
            for (i, token) in enumerate(sentence):
                atoken = self.tokenizer.tokenize(token)
                if len(atoken) == 0:
                    tokens.append('[UNK]')
                else:
                    tokens.append(atoken[0])
                in_sent.append(1)
                segment_ids.append(1)
                if i in offset2args:
                    args.append(self.role2id[offset2args[i]])
                else:
                    args.append(self.role2id['None'])
                
            tokens.append('[SEP]')
            args.append(self.role2id['None'])
            in_sent.append(0)
            segment_ids.append(1)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_masks = [1] * len(input_ids)
            while len(input_ids) < self.max_sentence_length:
                input_ids.append(0)
                input_masks.append(0)
                in_sent.append(0)
                segment_ids.append(0)
                args.append(self.role2id['None'])
            #labels = np.array(labels) - 169
            one_sample['input_ids'] = input_ids
            one_sample['input_masks'] = input_masks
            one_sample['in_sent'] = in_sent
            one_sample['segment_ids'] = segment_ids
            one_sample['args'] = args
            one_sample['args_offset'] = args_offset
            one_sample['gold_args'] = gold_args
            one_sample['trigger'] = event
            one_sample['ner'] = ner
            one_sample['sentence'] = sentence
            res.append(one_sample)
        return res




        
