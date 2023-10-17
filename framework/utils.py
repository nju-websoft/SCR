import torch
import numpy as np
import copy

def reset_id(labels, new_id):
    res = []
    for index in range(len(labels)):
        res.append(new_id[int(labels[index])])
    return torch.tensor(res)

def get_reset(event_list):
    new_id, id2label = {}, {}

    new_id[0] = torch.tensor(0)
    id2label[torch.tensor(0)] = 0
    for index, value in enumerate(event_list):
        new_id[value] = torch.tensor(index + 1)
        id2label[index+1] = value
    return new_id, id2label

def unpack_batch(sentence_ids, input_ids, input_masks, segment_ids, labels, ners, new_id, device):
    sentence_ids = torch.tensor(sentence_ids).to(device)
    input_ids = torch.tensor(np.array([item.cpu().detach().numpy() for item in input_ids])).to(device)
    input_masks = torch.tensor(np.array([item.cpu().detach().numpy() for item in input_masks])).to(device)
    segment_ids = torch.tensor(np.array([item.cpu().detach().numpy() for item in segment_ids])).to(device)
    ners = torch.tensor(np.array([item.cpu().detach().numpy() for item in ners])).to(device)
    if labels != None:
        if new_id != None:
            labels = torch.tensor(np.array([reset_id(item, new_id).cpu().detach().numpy() for item in labels])).to(device)
        else:
            labels = torch.tensor(np.array([item.cpu().detach().numpy() for item in labels])).to(device)
    return sentence_ids, input_ids, input_masks, segment_ids, labels, ners



# if one ins has two event, we should combine them as one ins 
def trigger_combine_event(old_data, new_data):
    if len(new_data) == 0:
        return old_data
    init = False
    res = []
    if len(old_data) == 0:
        init = True
        old_data = copy.deepcopy(new_data)
    for old_sample_index in range(len(old_data)-1, -1, -1):
        old_sample = old_data[old_sample_index]
        combine_flag = False
        for new_sample_index in range(len(new_data)-1, -1, -1):
            new_sample = new_data[new_sample_index]
            if old_sample['input_ids'] == new_sample['input_ids']:
                old_offset = torch.nonzero(torch.tensor(np.array(old_sample['labels'])))
                new_offset = torch.nonzero(torch.tensor(np.array(new_sample['labels'])))
                eqoffset = [int(val) for val in old_offset if val in new_offset]
                combine_flag = True
                if len(eqoffset) > 0:
                    eqflag = False
                    for i in eqoffset: 
                        if old_sample['labels'][i] != new_sample['labels'][i]:
                            # one ins has two event type on same trigger...
                            eqflag = True             
                    if eqflag == False:
                        new_data.remove(new_sample)
                    continue
                
                old_sample['labels'] = copy.deepcopy(list(np.array(old_sample['labels']) + np.array(new_sample['labels'])))
                new_data.remove(new_sample)
        if (combine_flag and init) or (init == False):
            temp = copy.deepcopy(old_sample)
            res.append(temp)
    res += new_data
    return res

def args_combine_event(old_data, new_data):
    if len(new_data) == 0:
        return old_data
    init = False
    res = []
    if len(old_data) == 0:
        init = True
        old_data = copy.deepcopy(new_data)
    for old_sample_index in range(len(old_data)-1, -1, -1):
        old_sample = old_data[old_sample_index]
        combine_flag = False
        for new_sample_index in range(len(new_data)-1, -1, -1):
            new_sample = new_data[new_sample_index]
            if old_sample['input_ids'] == new_sample['input_ids'] and old_sample['trigger'] == new_sample['trigger']:
                
                combine_flag = True
                if old_sample == new_sample:
                    new_data.remove(new_sample)
                    continue
                for i in range(len(old_sample['args'])):
                    if (old_sample['args'][i] == 0 and new_sample['args'][i] !=0) or (old_sample['args'][i] != 0 and new_sample['args'][i] ==0):
                        old_sample['args'][i] = old_sample['args'][i] + new_sample['args'][i]
                    elif old_sample['args'][i] != 0 and new_sample['args'][i] != 0 and new_sample['args'][i] != old_sample['args'][i]:
                        continue
                    elif old_sample['args'][i] != 0 and new_sample['args'][i] != 0 and new_sample['args'][i] == old_sample['args'][i]:
                        continue
                old_sample['gold_args'] = old_sample['gold_args'] + new_sample['gold_args']
                new_ner = [(start, end) for (start, end, _) in old_sample['gold_args']]

                
                old_sample['ner'] = old_sample['ner'] + new_ner
                old_sample['args_offset'] = list(set(old_sample['args_offset']+new_sample['args_offset']))
                
                new_data.remove(new_sample)
        if (combine_flag and init) or (init == False):
            temp = copy.deepcopy(old_sample)
            res.append(temp)
    res += new_data
    return res

