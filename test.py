# import string
import pandas as pd
import time
import os
import numpy as np
import logging
import argparse
from utils_metrics import get_entities_bio, f1_score, classification_report
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
import torch
import torch.nn as nn
import time
import math
import random
import torch.nn.functional as F
from seq2seq_model_add_MLP import Seq2SeqModel
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
# from inference import InputExample, prediction, cal_time
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
import warnings
warnings.filterwarnings("ignore")  # 忽略警告信息


def prediction(input_TXT, template_list, entity_dict, tokenizer, model, device, args, mlp=None, CM=None):
    input_TXT_list = input_TXT.split(' ')

    entity_list = []
    for i in range(len(input_TXT_list)):
        words = []
        for j in range(1, min(args.max_entity_length, len(input_TXT_list) - i + 1)):
            word = (' ').join(input_TXT_list[i:i+j])
            words.append([word])

        for w in words:
            entity = template_entity(w, input_TXT, i, template_list, entity_dict, tokenizer, model, device, args, mlp, CM) #[start_index,end_index,label,score, score_list]
            if entity[1] >= len(input_TXT_list):
                entity[1] = len(input_TXT_list)-1
            if entity[2] != 'O':
                entity_list.append(entity)
    i = 0
    if len(entity_list) > 1:
        while i < len(entity_list):
            j = i+1
            while j < len(entity_list):
                if (entity_list[i][1] < entity_list[j][0]) or (entity_list[i][0] > entity_list[j][1]):
                    j += 1
                else:
                    if entity_list[i][3] < entity_list[j][3]:
                        entity_list[i], entity_list[j] = entity_list[j], entity_list[i]
                        entity_list.pop(j)
                    else:
                        entity_list.pop(j)
            i += 1
    label_list = ['O'] * len(input_TXT_list)
    pred_logits = []

    for entity in entity_list:
        label_list[entity[0]:entity[1]+1] = ["I-"+entity[2]]*(entity[1]-entity[0]+1)
        label_list[entity[0]] = "B-"+entity[2]
        temp = []
        temp.append(input_TXT_list[entity[0]:entity[1]+1])
        temp.append(entity[2])
        temp.append(entity[3])
        temp.append(entity[4])
        pred_logits.append(temp)

    
    return label_list, pred_logits


def template_entity(words, input_TXT, start, template_list, entity_dict, tokenizer, model, device, args, mlp=None, CM=None):
    # input text -> template
    words_length = len(words)
    num_label = len(template_list)
    # words_length_list = [len(i) for i in words]
    input_TXT = [input_TXT]*(num_label*words_length)

    input_ids = tokenizer(input_TXT, return_tensors='pt')['input_ids']
    model.to(device)
    
    
    temp_list = []
    for i in range(words_length):
        for j in range(len(template_list)):
            temp_list.append(template_list[j].format(words[i]))

    output_ids = tokenizer(temp_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
    # output_ids[:, 0] = 2  # 为什么变成2
    output_length_list = [0]*num_label*words_length

    if args.te == 'te1':
        for i in range(len(temp_list)//num_label):
            base_length = ((tokenizer(temp_list[i * num_label], return_tensors='pt', padding=True, truncation=True)['input_ids']).shape)[1] - 3  # 为什么-4
            output_length_list[i*num_label:i*num_label+ num_label] = [base_length]*num_label
            output_length_list[i*num_label+num_label-1] += 1
    else:
        raise NotImplementedError
    
    if args.dataset == 'mit_m.10_shot':
        output_length_list[10] += 1
    elif args.dataset == 'MIT_MM':
        output_length_list[10] += 1
        output_length_list[17] += 1
    elif args.dataset == 'MIT_R':
        output_length_list[3] += 1

    score = [1]*num_label*words_length
    with torch.no_grad():
        origin_output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids[:, :output_ids.shape[1] - 2].to(device), output_hidden_states=True)
        output = origin_output[0]
        for i in range(output_ids.shape[1] - 2):  # 为什么-3
            # print(input_ids.shape)
            logits = output[:, i, :]
            logits = logits.softmax(dim=1)
            # values, predictions = logits.topk(1,dim = 1)
            logits = logits.to('cpu').numpy()
            # print(output_ids[:, i+1].item())
            for j in range(0, num_label*words_length):
                if i < output_length_list[j]:
                    score[j] = score[j] * logits[j][int(output_ids[j][i + 1])]

        # add mlp
        if args.use_mlp_or_not == True:
            mlp.to(device)
            input_mlp = origin_output.decoder_hidden_states[-1][0, (len(words[0].split())-1), :]
            mlp_logits = mlp(input_mlp)
            # without_O_mlp_logits = 0
            # O_mlp_logit = 0
            norm_score = (score-min(score))/(max(score)-min(score))
            norm_score = F.softmax(torch.tensor(norm_score)).numpy()
            if np.argmax(norm_score) != (norm_score.shape[0]-1):
                norm_score[:-1] = norm_score[:-1] * mlp_logits[1].item()
                norm_score[-1] *= mlp_logits[0].item()
            norm_score = norm_score.tolist()
        else:
            norm_score = score
        
        if args.use_cm_or_not == True and CM is not None:
            norm_score = torch.tensor(norm_score)
            norm_score = torch.mul(norm_score, CM)
            norm_score = norm_score.numpy().tolist()
            
    end = start + (len(words[0].split())-1)
    
        # score_list.append(score)
    # s_lst = norm_score[(norm_score.index(max(norm_score))//5)*5:(norm_score.index(max(norm_score))//5)*5+5]
    # norm_s_lst = (s_lst-min(s_lst))/(max(s_lst)-min(s_lst))
    return [start, end, entity_dict[(norm_score.index(max(norm_score))%num_label)], max(norm_score), norm_score]  # , mlp_logits.cpu().numpy().tolist()]  #[start_index,end_index,label,score,[score_list],[mlp_logits]]


def cal_time(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def test(examples, template_list, entity_dict, tokenizer, model, device, args, mlp=None, CM=None):
    trues_list = []
    preds_list = []
    logits_list = []
    string = ' '
    num_01 = len(examples)
    num_point = 0
    start = time.time()
    for example in examples:
        sources = string.join(example.words)
        words = sources.strip().split(' ')
        preds,logits = prediction(sources, template_list, entity_dict, tokenizer, model, device, args, mlp, CM)
        true = example.labels[0].strip().split(' ')
        assert len(words) == len(true) == len(preds)
        preds_list.append(preds)
        trues_list.append(true)
        print('%d/%d (%s)'%(num_point+1, num_01, cal_time(start)))
        print('Pred:', preds_list[num_point])
        print('Gold:', trues_list[num_point])
        num_point += 1
        # logits
        for i, l in enumerate(logits):
            temp = []
            for word in l[0]:
                temp.append(true[words.index(word)])
            logits[i].append(temp)
            logits_list.append(logits[i])

    true_entities = get_entities_bio(trues_list)
    pred_entities = get_entities_bio(preds_list)
    results = {
        "f1": f1_score(true_entities, pred_entities)
    }
    print(results["f1"])

    return logits_list


