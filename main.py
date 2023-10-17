import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from framework.utils import reset_id, get_reset, trigger_combine_event, unpack_batch
from framework.optimization import BertAdam, AdamW
from argparse import ArgumentParser
from model.trigger_encoder import triggerEncoder
from model.argument_detection import argumentDetection
from model.classifier import classifier
from model.entity_detection import entityDetection
from framework.config import Config
from framework.dataloader import *
from transformers import logging
from sklearn.cluster import KMeans
logging.set_verbosity_warning()
logging.set_verbosity_error()
import torch.nn.functional as F
import math
import warnings
warnings.filterwarnings('ignore')

def eval_trigger(trigger_encoder, trigger_classifier, eval_data, config, new_id, save, ltlabel, id2label):
    eval_data_loader = get_ACETriData_loader(eval_data, config, shuffle = True)
    trigger_encoder.eval()
    trigger_classifier.eval()
    pred_num = 0
    correct_num = 0
    label_num = 0
    pred_res = []
    for step, (sentence_ids, input_ids, input_masks, in_sent, segment_ids, labels, ners, sentence) in enumerate(eval_data_loader):
        
        sentence_ids, input_ids, input_masks, segment_ids, labels, ners = unpack_batch(sentence_ids, input_ids, input_masks, segment_ids, labels, ners, new_id, config.device)
        
        with torch.no_grad():
            feature = trigger_encoder(sentence_ids, input_ids, input_masks, segment_ids)
            #feature = torch.stack([x.to(device) for x in feature],dim=0)
            logits = trigger_classifier(feature, None, None)

        new_logits = logits
        for index, value in enumerate(in_sent):
            evetype = []
            pred_first = True
            value = value == 1
            gold_offset = torch.nonzero(labels[index][value]).squeeze(dim = 1)
            gold_label = torch.gather(labels[index][value], dim = 0, index = gold_offset)
            assert(len(gold_label) != 0)
            gold_label = [int(val) for val in gold_label]
            gold_offset = [int(val) for val in gold_offset]
            new_gold_label = []
            i = 0
            while i < len(gold_label):
                if i+1 >= len(gold_label):
                    if config.lttest and id2label[gold_label[i]] not in ltlabel:
                        break
                    else:
                        new_gold_label.append(gold_label[i])
                        break
                while gold_label[i] == gold_label[i+1] and gold_offset[i]+1 == gold_offset[i+1]:
                    i += 1
                    if i+1 >= len(gold_label):
                        break
                if config.lttest == False or id2label[gold_label[i]] in ltlabel:
                    new_gold_label.append(gold_label[i])
                i+=1
            gold_label = new_gold_label
            label_num += len(gold_label)
            
            res = new_logits[index][value,:]
            max_value, pred_tri_each_word = torch.max(res, 1)
            pred_trigger = 0
            offset = 0
            pred_offset, pred_label = [], []
            for offset, trigger in enumerate(pred_tri_each_word):
                if trigger!=0:
                    if config.lttest == False or id2label[int(trigger)] in ltlabel:
                        pred_offset.append(offset)
                        pred_label.append(trigger)
            new_pred_label = []
            i = 0
            while i < len(pred_label):
                if i+1 >= len(pred_label):
                    new_pred_label.append(pred_label[i])
                    break
                while pred_label[i] == pred_label[i+1] and pred_offset[i]+1 == pred_offset[i+1]:
                    i += 1
                    if i+1 >= len(pred_label):
                        break
                new_pred_label.append(pred_label[i])
                i+=1
            new_pred_label = [int(val) for val in new_pred_label]
            pred_num += len(new_pred_label)
            
            for pred_trigger in new_pred_label:
                if save:
                    if id2label[pred_trigger] not in evetype:
                        evetype.append(id2label[pred_trigger])
                        onesamp = {}
                        onesamp['sentence'] = sentence[index]
                        onesamp['trigger'] = id2label[pred_trigger]
                        onesamp['s_start'] = 0
                        pred_res.append(onesamp)
                if pred_trigger in gold_label:
                    correct_num += 1
                    gold_label.remove(pred_trigger)
    
    if pred_num == 0 or label_num == 0 or correct_num == 0:
        return 0
    pred_c = 100.0*correct_num/pred_num
    recall_c = 100.0*correct_num/label_num
    f1_c = 2*pred_c*recall_c/(pred_c+recall_c)
    if save:
        f = open(config.trigger_pred_file, 'w')
        json.dump(pred_res, f)
        f.close()
    return f1_c

def train_simple_trigger(trigger_encoder, trigger_classifier, tr_data, config, new_id):
    
    train_data_loader = get_ACETriData_loader(tr_data, config, shuffle = True)
    
    trigger_encoder.train()
    trigger_classifier.train()

    param_optimizer_1 = list(trigger_encoder.named_parameters())
    param_optimizer_1 = [n for n in param_optimizer_1 if 'pooler' not in n[0]]
    param_optimizer_2 = list(trigger_classifier.named_parameters())
    param_optimizer_2 = [n for n in param_optimizer_2 if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer_1
                    if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, "betas": (0.9, 0.999), 'lr':config.trigger_encoder_learning_rate},
        {'params': [p for n, p in param_optimizer_1
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, "betas": (0.9, 0.999),'lr':config.trigger_encoder_learning_rate},
        {'params': [p for n, p in param_optimizer_2
                    if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, "betas": (0.9, 0.999), 'lr':config.trigger_classifier_learning_rate},
        {'params': [p for n, p in param_optimizer_2
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, "betas": (0.9, 0.999), 'lr':config.trigger_classifier_learning_rate}
    ]
    optimizer = AdamW(params = optimizer_grouped_parameters)

    epoch_index, best_f1, es_index = 0, 0, 0
    fd_criterion = nn.CosineEmbeddingLoss()
    logits = None
    global_step = 0
    while(True):
        losses = []
        for step, (sentence_ids, input_ids, input_masks, in_sent, segment_ids, labels, ners, sentence) in enumerate(train_data_loader):
            sentence_ids, input_ids, input_masks, segment_ids, labels, ners = unpack_batch(sentence_ids, input_ids, input_masks, segment_ids, labels, ners, new_id, config.device)
            feature = trigger_encoder(sentence_ids, input_ids, input_masks, segment_ids)
            logits, loss = trigger_classifier(feature, input_masks, labels)
            losses.append(loss.cpu().detach().numpy())
            loss.backward()
            if (step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
        print(f"epoch: {epoch_index}, loss is {np.array(losses).mean()}")
        epoch_index += 1
        if epoch_index >= 5:
            break

def train_trigger(trigger_encoder, trigger_classifier, tr_data, de_data, seen_train_event, config, new_id, forward_encoder, forward_classifier, forward_event, trigger_tailed, ltlabel, id2label):
    

    if config.kd == True and forward_event != None:
        forward_index = reset_id(forward_event, new_id).cuda()
        print(forward_index)
        T = config.temp

    train_data_loader = get_ACETriData_loader(tr_data, config, shuffle = True)
    
    trigger_encoder.train()
    trigger_classifier.train()
    param_optimizer_1 = list(trigger_encoder.named_parameters())
    param_optimizer_1 = [n for n in param_optimizer_1 if 'pooler' not in n[0]]
    param_optimizer_2 = list(trigger_classifier.named_parameters())
    param_optimizer_2 = [n for n in param_optimizer_2 if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer_1
                    if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, "betas": (0.9, 0.999), 'lr':config.trigger_encoder_learning_rate},
        {'params': [p for n, p in param_optimizer_1
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, "betas": (0.9, 0.999),'lr':config.trigger_encoder_learning_rate},
        {'params': [p for n, p in param_optimizer_2
                    if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, "betas": (0.9, 0.999), 'lr':config.trigger_classifier_learning_rate},
        {'params': [p for n, p in param_optimizer_2
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, "betas": (0.9, 0.999), 'lr':config.trigger_classifier_learning_rate}
    ]
    if config.merit == 'epochs':
        num_train_optimization_steps = len(train_data_loader) // config.gradient_accumulation_steps * config.epochs
        optimizer = AdamW(params = optimizer_grouped_parameters, 
                            weight_decay=config.weight_decay)
    elif config.merit == 'early_stop':
        optimizer = AdamW(params = optimizer_grouped_parameters)

    epoch_index, best_f1, es_index = 0, 0, 0
    #fd_criterion = nn.CosineEmbeddingLoss(reduction = 'sum')
    fd_criterion = nn.CosineEmbeddingLoss()
    logits = None
    global_step = 0
    while(True):
        losses = []
        for step, (sentence_ids, input_ids, input_masks, in_sent, segment_ids, labels, ners, sentence) in enumerate(train_data_loader):
            sentence_ids, input_ids, input_masks, segment_ids, labels, ners = unpack_batch(sentence_ids, input_ids, input_masks, segment_ids, labels, ners, new_id, config.device)
            feature = trigger_encoder(sentence_ids, input_ids, input_masks, segment_ids)
            if len(trigger_tailed) != 0:
                tail_res = []
                for i, label in enumerate(labels):
                    flabels = label!=0
                    pos_labels = label[flabels]
                    pos_index = torch.nonzero(label)
                    for index, fe in enumerate(pos_labels):
                        if int(fe) in trigger_tailed:
                            protos, standard = trigger_tailed[int(fe)]
                            protos = protos[flabels]
                            standard = standard[flabels]
                            for st in range(len(standard)):
                                s = torch.tensor(np.random.normal(0, standard[st], 1)).cuda()
                                j = pos_index[index]
                                feature[i][j] += s
                                tail_res.append((i,j,s))

            logits, loss = trigger_classifier(feature, input_masks, labels)
            

            if config.kd == True and forward_event != None:
                #print(tail_res)
                kd_loss = 0
                temp_masks = copy.deepcopy(input_masks)
                forward_features = forward_encoder(sentence_ids, input_ids, temp_masks, segment_ids)
                if len(trigger_tailed) != 0:
                    for i,j,s in tail_res:
                        forward_features[i][j] += s
                forward_logits = forward_classifier(forward_features, temp_masks, None)
                forward_logits = (forward_logits.index_select(2, forward_index)/T).view(-1, len(forward_event))
                new_logits = (logits.index_select(2, forward_index)/T).view(-1, len(forward_event))
                active_loss = (input_masks.view(-1) == 1).cuda()
                forward_logits = forward_logits[active_loss]
                new_logits = new_logits[active_loss]


                if config.select == True:
                    max_forward_index = max(forward_index)
                    label_index = (labels.view(-1)<=max_forward_index)[active_loss].cuda()
                    forward_logits[:,0] = 0
                    new_logits[:,0] = 0
                    forward_logits = forward_logits[label_index]
                    new_logits = new_logits[label_index]

                    forward_logits = F.softmax(forward_logits, dim = 1)
                    new_logits = F.log_softmax(new_logits, dim = 1)
                    kd_loss = -torch.mean(torch.sum(forward_logits * new_logits, dim = 1))
                    #kd_loss = -torch.sum(torch.sum(forward_logits * new_logits, dim = 1))

                if config.attention == True:
                    attention = trigger_encoder.get_attention(input_ids, input_masks, segment_ids)
                    forward_attention = forward_encoder.get_attention(input_ids, input_masks, segment_ids)
                    attention = attention.matmul(feature)
                    forward_attention = forward_attention.matmul(forward_features)
                    attention = F.normalize(attention, p=2, dim=2).view(-1, attention.shape[2])[active_loss]
                    forward_attention = F.normalize(forward_attention, p=2, dim=2).view(-1, forward_attention.shape[2])[active_loss]
                    fd_loss = fd_criterion(attention, forward_attention, torch.ones(attention.shape[0]).cuda())
                    kd_loss = kd_loss + fd_loss

                loss = (1-config.alpha)*loss+config.alpha*kd_loss
            losses.append(loss.cpu().detach().numpy())
            loss.backward()
            if (step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
        if  config.merit == 'early_stop':
            res = 0
            res = eval_trigger(trigger_encoder, trigger_classifier, de_data, config, new_id, False, ltlabel, id2label)
            trigger_encoder.train()
            trigger_classifier.train()
            if res > best_f1:
                best_f1 = res
                es_index = 0
                encoder_output_path = config.output_dir+ config.trigger_encoder_file
                torch.save(trigger_encoder.state_dict(), encoder_output_path)
                classifier_output_path = config.output_dir+ config.trigger_classifier_file
                torch.save(trigger_classifier.state_dict(), classifier_output_path)
            else:
                es_index += 1
            print(f"epoch: {epoch_index}, loss is {np.array(losses).mean()}, f1 is {res} and best f1 is {best_f1}")
            epoch_index += 1
            if es_index >= config.early_stop:
                trigger_encoder.load_state_dict(torch.load(encoder_output_path))
                trigger_classifier.load_state_dict(torch.load(classifier_output_path))
                break
        if  config.merit == 'epochs':
            print(f"epoch: {epoch_index}, loss is {np.array(losses).mean()}")
            epoch_index += 1
            if epoch_index >= config.epochs:
                break

def select_data(config, trigger_encoder, relation_dataset, new_id, event):
    train_data_loader = get_ACETriData_loader(relation_dataset, config, shuffle = False, batch_size = 1)
    features = []
    trigger_encoder.eval()
    for step, (sentence_ids, input_ids, input_masks, in_sent, segment_ids, labels, ners, sentence) in enumerate(train_data_loader):
        sentence_ids, input_ids, input_masks, segment_ids, labels, ners = unpack_batch(sentence_ids, input_ids, input_masks, segment_ids, labels, ners, new_id, config.device)
        with torch.no_grad():
            feature = trigger_encoder.get_feature(sentence_ids, input_ids, input_masks, segment_ids).cpu()
            
        features.append(feature)
    features = np.concatenate(features)
    num_clusters = min(config.memory_size, len(relation_dataset))
    if num_clusters == len(relation_dataset):
        memory = []
        for i in relation_dataset:
            memory.append(i)
        return memory
    distances = KMeans(n_clusters = num_clusters, random_state = 0).fit_transform(features)
    memory = []
    for k in range(num_clusters):
        select_index = np.argmin(distances[:, k])
        ins = relation_dataset[select_index]
        memory.append(ins)
    return memory

def addPseudoLabel(trigger_encoder, trigger_classifier, data, config, id2label):
    pseudo_data = []
    eval_data_loader = get_ACETriData_loader(data, config, shuffle = True, batch_size = 1)
    trigger_encoder.eval()
    trigger_classifier.eval()
    for step, (sentence_ids, input_ids, input_masks, in_sent, segment_ids, labels, ners, sentence) in enumerate(eval_data_loader):
        sentence_ids, input_ids, input_masks, segment_ids, labels, ners = unpack_batch(sentence_ids, input_ids, input_masks, segment_ids, labels, ners, None, config.device)
        with torch.no_grad():
            feature = trigger_encoder(sentence_ids, input_ids, input_masks, segment_ids)
            logits = trigger_classifier(feature, None, None)

        new_logits = logits
        for index, value in enumerate(in_sent):
            pred_first = True
            value = value == 1
            gold_offset = torch.nonzero(labels[index][value]).squeeze(dim = 1)
            gold_label = torch.gather(labels[index][value], dim = 0, index = gold_offset)
            gold_label = [int(val) for val in gold_label]
            gold_offset = [int(val) for val in gold_offset]
            res = new_logits[index][value,:]
            max_value, pred_tri_each_word = torch.max(res, 1)
            pred_trigger = 0
            for offset, trigger in enumerate(pred_tri_each_word):
                if trigger!=0 and max_value[offset] > 0.8 and offset not in gold_offset:
                    one_sample = {}
                    one_sample['sentence_ids'] = sentence_ids[0].tolist()
                    one_sample['input_ids'] = input_ids[0].tolist()
                    one_sample['input_masks'] = input_masks[0].tolist()
                    pseudo_label = torch.zeros(len(input_ids[0]))
                    pseudo_label[offset] = id2label[int(trigger)]           
                    one_sample['labels'] = pseudo_label.tolist()
                    one_sample['in_sent'] = in_sent[0].tolist()
                    one_sample['segment_ids'] = segment_ids[0].tolist()
                    one_sample['ners'] = ners[0].tolist()
                    one_sample['sentence'] = sentence[0]
                    pseudo_data.append(one_sample)
    return pseudo_data + data

def get_trigger_proto(config, trigger_encoder, relation_dataset, new_id, event):
    train_data_loader = get_ACETriData_loader(relation_dataset, config, shuffle = False, batch_size = 1)
    features = []
    trigger_encoder.eval()
    for step, (sentence_ids, input_ids, input_masks, in_sent, segment_ids, labels, ners, sentence) in enumerate(train_data_loader):
        sentence_ids, input_ids, input_masks, segment_ids, labels, ners = unpack_batch(sentence_ids, input_ids, input_masks, segment_ids, labels, ners, new_id, config.device)
        with torch.no_grad():
            feature = trigger_encoder(sentence_ids, input_ids, input_masks, segment_ids)
            feature = feature[labels == event]
        features.append(feature)
    features = torch.cat(features, dim = 0)
    proto = torch.mean(features, dim = 0, keepdim = True).cpu()
    standard = torch.sqrt(torch.var(features, dim=0)).cpu()
    return proto, standard

def kt_long_tailed(trigger_protos, trigger_num):
    len_tail = int(0.8*len(trigger_num))
    res = {}
    for i in range(len_tail):
        tail_event = trigger_num[i][0]
        tail_proto, tail_standard = trigger_protos[tail_event]
        tail_proto = tail_proto.squeeze(0)
        tail_standard = tail_standard.squeeze(0)
        tail_cos, all_proto, all_standard = [], [], []
        for event, (proto, standard) in trigger_protos.items():
            proto = proto.squeeze(0)
            standard = standard.squeeze(0)
            if event != tail_event:
                tail_cos.append(F.cosine_similarity(tail_proto, proto, dim = 0))
                all_proto.append(proto)
                all_standard.append(standard)
        all_proto = torch.stack(all_proto)
        all_standard = torch.stack(all_standard)
        tail_cos = torch.stack(tail_cos)
        tail_cos = F.softmax(tail_cos, dim=0)
        res_standard = torch.matmul(tail_cos, all_standard)
        res_proto = torch.matmul(tail_cos, all_proto)
        res[tail_event] = (res_proto, res_standard)
    return res

def eval_entity_detection(entity_detection, eval_data, config, new_id):
    eval_data_loader = get_ACETriData_loader(eval_data, config, shuffle = True)
    entity_detection.eval()
    pred_num = 0
    correct_num = 0
    label_num = 0
    pred_res = []
    for step, (sentence_ids, input_ids, input_masks, in_sent, segment_ids, labels, ners, sentence) in enumerate(eval_data_loader):
        
        sentence_ids, input_ids, input_masks, segment_ids, labels, ners = unpack_batch(sentence_ids, input_ids, input_masks, segment_ids, labels, ners, new_id, config.device)
        
        with torch.no_grad():
            logits = entity_detection.get_res(input_ids, segment_ids, input_masks)

        new_logits = logits

        for index, value in enumerate(in_sent):
            value = value == 1
            pred_logits = torch.tensor(new_logits[index])[1:-1].tolist()
            gold_offset = []
            start, end, now = 0,0,0
            for offset, wo in enumerate(ners[index][value]):
                wo = int(wo)
                if wo !=0 and now == 0:
                    now = wo
                    start = offset
                    end = offset+1
                elif wo !=0 and now !=0 and wo == now:
                    end = offset+1
                elif wo !=0 and now !=0 and wo != now:
                    now = wo
                    gold_offset.append((start, end))
                    start = offset
                    end = offset+1
                elif wo == 0 and now == 0:
                    start, end = 0, 0
                elif wo == 0 and now != 0:
                    now = 0
                    gold_offset.append((start, end))
            if now != 0:
                gold_offset.append((start, end))
            
            for i in gold_offset:
                start, end = i
                for j in range(start, end-1):
                    if ners[index][value][j] != ners[index][value][j+1]:
                        print(ners[index][value])
                        print(gold_offset)
                        assert(0)
            label_num+=len(gold_offset)

            
            pred_offset = []
            start, end, now = 0,0,0
            pred_tri_each_word = pred_logits
            for offset, wo in enumerate(pred_tri_each_word):
                wo = int(wo)
                if wo !=0 and now == 0:
                    now = wo
                    start = offset
                    end = offset+1
                elif wo !=0 and now !=0 and wo == now:
                    end = offset+1
                elif wo !=0 and now !=0 and wo != now:
                    now = wo
                    pred_offset.append((start, end))
                    start = offset
                    end = offset+1
                elif wo == 0 and now == 0:
                    start, end = 0, 0
                elif wo == 0 and now != 0:
                    now = 0
                    pred_offset.append((start, end))
            if now != 0:
                pred_offset.append((start, end))

            pred_num += len(pred_offset)
            

            for pred in pred_offset:
                if pred in gold_offset:
                    correct_num += 1
            

    if pred_num == 0 or label_num == 0 or correct_num == 0:
        return 0
    pred_c = 100.0*correct_num/pred_num
    recall_c = 100.0*correct_num/label_num
    f1_c = 2*pred_c*recall_c/(pred_c+recall_c)
    return f1_c

def pred_entity_detection(config, entity_detection, sampler):
    eval_data = sampler.read_pred_sample(config.trigger_pred_file)
    eval_data_loader = get_ACEPredData_loader(eval_data, config, shuffle = True)
    entity_detection.eval()
    pred_num = 0
    correct_num = 0
    label_num = 0
    pred_res = []
    for step, (input_ids, input_masks, in_sent, segment_ids, sentence, event) in enumerate(eval_data_loader):
        
        input_ids = torch.tensor(np.array([item.cpu().detach().numpy() for item in input_ids])).cuda()
        input_masks = torch.tensor(np.array([item.cpu().detach().numpy() for item in input_masks])).cuda()
        segment_ids = torch.tensor(np.array([item.cpu().detach().numpy() for item in segment_ids])).cuda()
        with torch.no_grad():
            logits = entity_detection.get_res(input_ids, segment_ids, input_masks)

        new_logits = logits

        for index, value in enumerate(in_sent):
            value = value == 1
            pred_logits = torch.tensor(new_logits[index])[1:-1].tolist()
            pred_offset = []
            start, end, now = 0,0,0
            pred_tri_each_word = pred_logits
            for offset, wo in enumerate(pred_tri_each_word):
                wo = int(wo)
                if wo !=0 and now == 0:
                    now = wo
                    start = offset
                    end = offset+1
                elif wo !=0 and now !=0 and wo == now:
                    end = offset+1
                elif wo !=0 and now !=0 and wo != now:
                    now = wo
                    pred_offset.append((start, end))
                    start = offset
                    end = offset+1
                elif wo == 0 and now == 0:
                    start, end = 0, 0
                elif wo == 0 and now != 0:
                    now = 0
                    pred_offset.append((start, end))
            if now != 0:
                pred_offset.append((start, end))

            onesamp = {}
            onesamp['sentence'] = sentence[index]
            onesamp['trigger'] = event[index]
            onesamp['s_start'] = 0
            onesamp['ner'] = pred_offset
            pred_res.append(onesamp)

            
            
    f = open(config.entity_pred_file, 'w')
    json.dump(pred_res, f)
    f.close()
    print('Entity predict over')
    
def train_entity_detection(entity_detection, tr_data, de_data, config, new_id):
    train_data_loader = get_ACETriData_loader(tr_data, config, shuffle = True)
    
    entity_detection.train()

    param_optimizer_1 = list(entity_detection.named_parameters())
    param_optimizer_1 = [n for n in param_optimizer_1 if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer_1
                    if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, "betas": (0.9, 0.999), 'lr':config.entity_detection_leraning_rate},
        {'params': [p for n, p in param_optimizer_1
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, "betas": (0.9, 0.999),'lr':config.entity_detection_leraning_rate}
    ]
    optimizer = AdamW(params = optimizer_grouped_parameters)

    epoch_index, best_f1, es_index = 0, 0, 0
    fd_criterion = nn.CosineEmbeddingLoss()
    logits = None
    global_step = 0
    while(True):
        losses = []
        for step, (sentence_ids, input_ids, input_masks, in_sent, segment_ids, labels, ners, sentence) in enumerate(train_data_loader):
            sentence_ids, input_ids, input_masks, segment_ids, labels, ners = unpack_batch(sentence_ids, input_ids, input_masks, segment_ids, labels, ners, new_id, config.device)
            loss = entity_detection(input_ids, ners, segment_ids, input_masks)
            losses.append(loss.cpu().detach().numpy())
            loss.backward()
            if (step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
        res = 0
        res = eval_entity_detection(entity_detection, de_data, config, new_id)
        entity_detection.train()
        if res > best_f1:
            best_f1 = res
            es_index = 0
            encoder_output_path = config.output_dir+ config.entity_file
            torch.save(entity_detection.state_dict(), encoder_output_path)
        else:
            es_index += 1
        print(f"epoch: {epoch_index}, loss is {np.array(losses).mean()}, f1 is {res} and best f1 is {best_f1}")
        epoch_index += 1
        if es_index >= config.early_stop:
            entity_detection.load_state_dict(torch.load(encoder_output_path))
            break

def train_argument_detection(argument_detection, tr_data, de_data, config, metadata, unseen_metadata):
    train_data_loader = get_ACEArgData_loader(tr_data, config, shuffle = True)
    
    argument_detection.train()

    param_optimizer_1 = list(argument_detection.named_parameters())
    param_optimizer_1 = [n for n in param_optimizer_1 if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer_1
                    if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, "betas": (0.9, 0.999), 'lr':config.argument_detection_leraning_rate},
        {'params': [p for n, p in param_optimizer_1
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, "betas": (0.9, 0.999),'lr':config.argument_detection_leraning_rate}
    ]
    optimizer = AdamW(params = optimizer_grouped_parameters)

    epoch_index, best_f1, es_index = 0, 0, 0
    fd_criterion = nn.CosineEmbeddingLoss()
    logits = None
    global_step = 0
    while(True):
        losses = []
        for step, (sentence, input_ids, input_masks, in_sent, segment_ids, args, args_offset, gold_args, ner, trigger) in enumerate(train_data_loader):
            input_ids = torch.tensor(np.array([item.cpu().detach().numpy() for item in input_ids])).cuda()
            input_masks = torch.tensor(np.array([item.cpu().detach().numpy() for item in input_masks])).cuda()
            segment_ids = torch.tensor(np.array([item.cpu().detach().numpy() for item in segment_ids])).cuda()
            args = torch.tensor(np.array([item.cpu().detach().numpy() for item in args])).cuda()

            loss = argument_detection(input_ids, args, segment_ids, input_masks, args_offset,  metadata, unseen_metadata, trigger, ner, gold_args)
            losses.append(loss.cpu().detach().numpy())
            loss.backward()
            if (step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
        res = 0
        res = eval_argument_detection(argument_detection, de_data, config, metadata)
        argument_detection.train()
        if res > best_f1:
            best_f1 = res
            es_index = 0
            encoder_output_path = config.output_dir+ config.argument_file
            torch.save(argument_detection.state_dict(), encoder_output_path)
        else:
            es_index += 1
        print(f"epoch: {epoch_index}, loss is {np.array(losses).mean()}, f1 is {res} and best f1 is {best_f1}")
        epoch_index += 1
        if es_index >= config.early_stop:
            argument_detection.load_state_dict(torch.load(encoder_output_path))
            break

def eval_argument_detection(argument_detection, eval_data, config, metadata):
    eval_data_loader = get_ACEArgData_loader(eval_data, config, shuffle = True)
    argument_detection.eval()
    pred_num = 0
    correct_num = 0
    label_num = 0
    pred_res = []
    for step, (sentence, input_ids, input_masks, in_sent, segment_ids, args, args_offset, gold_args, ner, trigger) in enumerate(eval_data_loader):
        
        input_ids = torch.tensor(np.array([item.cpu().detach().numpy() for item in input_ids])).cuda()
        input_masks = torch.tensor(np.array([item.cpu().detach().numpy() for item in input_masks])).cuda()
        segment_ids = torch.tensor(np.array([item.cpu().detach().numpy() for item in segment_ids])).cuda() 

        with torch.no_grad():
            logits = argument_detection.get_res(input_ids, segment_ids, input_masks, ner)
        
        for i in range(len(in_sent)):
            new_logits = logits[i]
            seen_args = copy.deepcopy(metadata[trigger[i]])
            seen_args += [0]
            pred_roles = []
            if new_logits == None:
                continue
            
            for index, value in enumerate(new_logits):
                logi = value[seen_args]
                max_value, pred_role = torch.max(logi, dim = 0)
                start, end = ner[i][index]
                one_pred = (start, end, seen_args[int(pred_role)])
                if seen_args[int(pred_role)] != 0:
                    pred_roles.append(one_pred)
            one_gold_args = copy.deepcopy(gold_args[i])
            pred_num += len(pred_roles)
            label_num += len(one_gold_args)
            for preds in pred_roles:
                if preds in one_gold_args:
                    correct_num += 1
                    one_gold_args.remove(preds)

    if pred_num == 0 or label_num == 0 or correct_num == 0:
        return 0
    pred_c = 100.0*correct_num/pred_num
    recall_c = 100.0*correct_num/label_num
    f1_c = 2*pred_c*recall_c/(pred_c+recall_c)
    return f1_c

def pred_argument_detection(config, argument_detection, sampler, metadata, gold_data):
    eval_data = sampler.read_pred_ner_sample(config.entity_pred_file)
    eval_data_loader = get_ACEPredNerData_loader(eval_data, config, shuffle = True)
    argument_detection.eval()
    pred_num = 0
    correct_num = 0
    label_num = 0
    pred_res = []
    gold_args = {}
    gold_data_loader = get_ACEArgData_loader(gold_data, config, shuffle = True, batch_size = 1)
    for step, (sentence, _, _, _, _, args, args_offset, gold, _, trig) in enumerate(gold_data_loader):
        sentence = copy.deepcopy(sentence[0])
        trig = copy.deepcopy(trig[0])
        gold = copy.deepcopy(gold[0])
        sentence = ''.join(sentence) + str(trig)
        if sentence in gold_args:
            print(gold_args[sentence])
            print(gold)
            assert(0)
        gold_args[sentence] = gold
        label_num += len(gold)
    for step, (input_ids, input_masks, in_sent, segment_ids, sentence, trigger, ner) in enumerate(eval_data_loader):
        
        input_ids = torch.tensor(np.array([item.cpu().detach().numpy() for item in input_ids])).cuda()
        input_masks = torch.tensor(np.array([item.cpu().detach().numpy() for item in input_masks])).cuda()
        segment_ids = torch.tensor(np.array([item.cpu().detach().numpy() for item in segment_ids])).cuda() 

        with torch.no_grad():
            logits = argument_detection.get_res(input_ids, segment_ids, input_masks, ner)
        
        for i in range(len(in_sent)):
            sent = copy.deepcopy(sentence[i])
            tr = copy.deepcopy(trigger[i])
            tr = sampler.index2vocab[tr]
            sent = ''.join(sent) + str(tr)
            new_logits = logits[i]
            seen_args = copy.deepcopy(metadata[tr])
            seen_args += [0]
            pred_roles = []
            if new_logits == None:
                continue
            
            for index, value in enumerate(new_logits):
                logi = value[seen_args]
                max_value, pred_role = torch.max(logi, dim = 0)
                start, end = ner[i][index]
                one_pred = (start, end, seen_args[int(pred_role)])
                if seen_args[int(pred_role)] != 0:
                    pred_roles.append(one_pred)
            
            if sent in gold_args:
                one_gold_args = copy.deepcopy(gold_args[sent])
                pred_num += len(pred_roles)
                for preds in pred_roles:
                    if preds in one_gold_args:
                        while(preds in one_gold_args):
                            correct_num += 1
                            one_gold_args.remove(preds)
            else:
                pred_num += len(pred_roles)
           

    if pred_num == 0 or label_num == 0 or correct_num == 0:
        return 0
    pred_c = 100.0*correct_num/pred_num
    recall_c = 100.0*correct_num/label_num
    f1_c = 2*pred_c*recall_c/(pred_c+recall_c)
    return f1_c

def select_argu_data(config, argument_detection, relation_dataset,new_id, event_mention):
    train_data_loader = get_ACEArgData_loader(relation_dataset, config, shuffle = False, batch_size = 1)
    features = []
    argument_detection.eval()
    for step, (sentence, input_ids, input_masks, in_sent, segment_ids, args, args_offset, gold_args, ner, trigger) in enumerate(train_data_loader):
        input_ids = torch.tensor(np.array([item.cpu().detach().numpy() for item in input_ids])).cuda()
        input_masks = torch.tensor(np.array([item.cpu().detach().numpy() for item in input_masks])).cuda()
        segment_ids = torch.tensor(np.array([item.cpu().detach().numpy() for item in segment_ids])).cuda() 
        with torch.no_grad():
            feature = argument_detection.get_feature(input_ids, segment_ids, input_masks).cpu()
            
        features.append(feature)
    features = np.concatenate(features)
    num_clusters = min(config.memory_size, len(relation_dataset))
    if num_clusters == len(relation_dataset):
        memory = []
        for i in relation_dataset:
            memory.append(i)
        return memory
    distances = KMeans(n_clusters = num_clusters, random_state = 0).fit_transform(features)
    memory = []
    for k in range(num_clusters):
        select_index = np.argmin(distances[:, k])
        ins = relation_dataset[select_index]
        memory.append(ins)
    return memory

def main():
    # load config 

    

    parser = ArgumentParser()
    parser.add_argument('--config', default='./config/ace.ini')
    args = parser.parse_args()
    config = Config(args.config)
    

    # set train param
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size_per_step = int(config.batch_size / config.gradient_accumulation_steps)

    triger_result_total, trigger_result_cur, argument_result_total, argument_result_cur = [], [], [], []
    # six truns and get average
    for i in range(config.total_round):
        print(f"Now is round {i}")
        config.seed += 100
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        # now is trigger detection task
        sampler = ACETriDataloder(config, i)
        trigger_one_round_res = []
        argument_one_round_res = []

        # trigger memory space
        trigger_memorized_samples = {}

        # argument memory space
        argument_memorized_samples = {}

        # init trigger encode model
        entity_detection = entityDetection(config).to(config.device)
        argument_detection = argumentDetection(config).to(config.device)
        trigger_encoder = triggerEncoder(config).to(config.device)
        
        forward_encoder, forward_classifier = None, None
        forward_event, forward_num = None, None
        trigger_num = {}

        # task1 to task5
        for steps, (tr_data, de_data, cur_te_data, current_event, seen_event, total_te_data, train_data_dict, total_dev_data, 
                    tr_args_data, de_args_data, cur_args_te_data, seen_test_args_data, seen_dev_args_data, train_args_data_dict) in enumerate(sampler):
            # get memory data
            
            cur_event_name = [sampler.index2vocab[x] for x in current_event]
            seen_event_name = [sampler.index2vocab[x] for x in seen_event]
            
            
            print(cur_event_name)
            new_id, id2label = get_reset(seen_event)
            now_num = len(seen_event_name)



            trigger_classifier = classifier(config, len(seen_event)+1).to(config.device)
            print('now is training trigger detection model')
            trigger_protos, trigger_tailed = {}, {}
            
            
           

            if config.longtailedkt:
                for event in cur_event_name:
                    event_mention = int(new_id[sampler.vocab2index[event]])
                    trigger_num[event_mention] = len(train_data_dict[event])
            
            if steps!=0:
                config.alpha = forward_num / now_num
                if config.pseudo == True:
                    tr_data = addPseudoLabel(forward_encoder, forward_classifier, tr_data, config, id2label)
                    tr_data = trigger_combine_event([], tr_data) 
                if config.longtailedkt:
                    train_simple_trigger(trigger_encoder, trigger_classifier, tr_data, config, new_id)
                    # get protos
                    for event in cur_event_name:
                        event_mention = int(new_id[sampler.vocab2index[event]])
                        proto, standard = get_trigger_proto(config, trigger_encoder, train_data_dict[event], new_id, event_mention)
                        #print(proto, standard)
                        trigger_protos[event_mention] = (proto, standard)
                        
                    for event in seen_event_name:
                        if event not in cur_event_name:
                            event_mention = int(new_id[sampler.vocab2index[event]])
                            proto, standard = get_trigger_proto(config, trigger_encoder, trigger_memorized_samples[event], new_id, event_mention)
                            trigger_protos[event_mention] = (proto, standard)

                    sorted_trigger_num = sorted(trigger_num.items(), key = lambda kv:(kv[1], kv[0]))
                    trigger_tailed = kt_long_tailed(trigger_protos, sorted_trigger_num)

            
            # add trigger memory data:
            if config.memory:
                trigger_memory_data = []            
                for event in seen_event_name:
                    if event not in cur_event_name:
                        trigger_memory_data += trigger_memorized_samples[event]
                trigger_memory_data = trigger_combine_event([], trigger_memory_data)
                tr_data = trigger_combine_event(tr_data, trigger_memory_data)
            
            train_trigger(trigger_encoder, trigger_classifier, tr_data, total_dev_data, seen_event, config, new_id, forward_encoder, forward_classifier, forward_event, trigger_tailed, sampler.ltlabel, id2label)
            

            # get forward event
            forward_event =  copy.deepcopy(seen_event)
            forward_event.append(0)
            forward_encoder, forward_classifier = copy.deepcopy(trigger_encoder), copy.deepcopy(trigger_classifier)
            
            forward_num = now_num
            torch.cuda.empty_cache()

            
            if config.memory:
                # get memory data
                for event in cur_event_name:
                    event_mention = int(new_id[sampler.vocab2index[event]])
                    trigger_memorized_samples[event] = select_data(config, trigger_encoder, train_data_dict[event],new_id, event_mention)

            
            # add new->old pseudo label:
            if steps!=0 and config.pseudo == True:
                for event, value in trigger_memorized_samples.items():
                    mem_data = value
                    new_mem_data = addPseudoLabel(trigger_encoder, trigger_classifier, mem_data, config, id2label)
                    new_mem_data = trigger_combine_event([], new_mem_data)
                    trigger_memorized_samples[event] = new_mem_data
            
            

            


            test_f1 = eval_trigger(trigger_encoder, trigger_classifier, total_te_data, config, new_id, True, sampler.ltlabel, id2label)

                

            print(f"Now task is {steps+1}, total trigger f1 is {test_f1}")
            trigger_one_round_res.append(test_f1)
            if config.argument:
                print('now is training entity detection model')
                train_entity_detection(entity_detection, tr_data, total_dev_data, config, new_id)
                pred_entity_detection(config, entity_detection, sampler)
                print('now is training argument detection model')
                if config.memory:
                    # add argument memory data:
                    argument_memory_data = []            
                    for event in seen_event_name:
                        if event not in cur_event_name:
                            argument_memory_data += argument_memorized_samples[event]
                    argument_memory_data = args_combine_event([], argument_memory_data)
                    tr_args_data = args_combine_event(tr_args_data, argument_memory_data)

                train_argument_detection(argument_detection, tr_args_data, seen_dev_args_data, config, sampler.metadata, sampler.unseen_metadata)

                if config.memory:
                    # get memory data
                    for event in cur_event_name:
                        event_mention = int(new_id[sampler.vocab2index[event]])
                        argument_memorized_samples[event] = select_argu_data(config, argument_detection, train_args_data_dict[event],new_id, event_mention)

                f1 = pred_argument_detection(config, argument_detection, sampler, sampler.metadata, seen_test_args_data)
                print(f"Now task is {steps+1}, total argument f1 is {f1}")
                argument_one_round_res.append(f1)

        triger_result_total.append(trigger_one_round_res)
        argument_result_total.append(argument_one_round_res)
        
    triger_result_total = np.array(triger_result_total)
    print(triger_result_total.mean(axis=0))
    if config.argument:
        argument_result_total = np.array(argument_result_total)
        print(argument_result_total.mean(axis=0))

if __name__ == "__main__":
    main()
