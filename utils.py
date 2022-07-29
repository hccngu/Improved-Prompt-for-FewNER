# import string
from calendar import day_name
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
from test import template_entity
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


class InputExample():
    def __init__(self, words, labels):
        self.words = words
        self.labels = labels


class Trainingset_InputExample():
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs


def train_prediction(input_words, output_words, template_list, entity_dict, tokenizer, model, device, args):
    # entity_words = output_words[:output_words.index('is')]
    is_index = len(output_words) - output_words[::-1].index('is') - 1
    entity_words = ' '.join(output_words[: is_index])
    print("entity_words:", entity_words)
    print("input_words:", input_words)
    print("output_words:", output_words)
    entity = template_entity([entity_words], ' '.join(input_words), 0, template_list, entity_dict, tokenizer, model, device, args)
    label_word = output_words[-2]
    for i, tl in enumerate(template_list):
        if label_word in tl.split():
            label = i
            break

    return entity[4], label


def Sample(train_df, num, random_state, te):
    if random_state != -1:
        random.seed(random_state)
    start = random.randint(0, 30000)
    temp = {}
    num_list = []
    for i in range(start, train_df.shape[0]):

        label_words = train_df['target_text'][i].strip().split(' ')

        if (te == 'te1') or (te == 'te3'):
            label = label_words[-2]
        elif (te == 'te2') or (te == 'te4'):
            label = label_words[-1]
        else:
            raise NotImplementedError

        if label in ['named', 'entity', 'none']:
            num_list.append(i)
            if label in temp:
                temp[label] += 1
            else:
                temp[label] = 1
        else:
            if label in temp:
                if temp[label] >= num:
                    continue
                num_list.append(i)
                temp[label] += 1
            else:
                num_list.append(i)
                temp[label] = 1
        
        flag = True
        for k in temp:
            if temp[k] < num:
                flag = False
                break
        if flag == True:
            if len(temp) == 5:
                break

    print(temp)
    return train_df.loc[num_list, ["input_text", "target_text"]]


def set_template(dataset_name, te):
    if dataset_name == 'cnoll2003':
        entity_dict = {0: 'LOC', 1: 'PER', 2: 'ORG', 3: 'MISC', 4: 'O'}
        if te == 'te1':
            template_list = ["{} is a location entity", "{} is a person entity", "{} is an organization entity",
                                "{} is an other entity", "{} is not a named entity"]
        elif te == 'te2':
            template_list = ["The entity type of {} is location","The entity type of {} is person",
                                "The entity type of {} is organization","The entity type of {} is other","The entity type of {} is none entity"]
        elif te == 'te3':
            template_list = ["{} belongs to location category",
                            " {} belongs to person category", 
                            " {} belongs to organization category",
                            " {} belongs to other category", 
                            " {} belongs to none category"]
        elif te == 'te4':
            template_list = ["{} should be tagged as location",
                            "{} should be tagged as person", 
                            "{} should be tagged as organization",
                            "{} should be tagged as other", 
                            "{} should tagged as none entity"]
        else:
            raise NotImplementedError
    elif dataset_name == 'MIT_M':
        entity_dict = {0: "ACTOR", 1: "YEAR", 2: "TITLE", 3: "GENRE", 4: "DIRECTOR", 5: "SONG", 6: "PLOT", 7: "REVIEW", 8: "CHARACTER", 9: "RATING", 10: "RATINGS_AVERAGE", 11: "TRAILER", 12: "O"}
        if te == 'te1':
            template_list = ["{} is an actor entity",
                             "{} is a year entity", 
                             "{} is a title entity",
                             "{} is a genre entity", 
                             "{} is a director entity", 
                             "{} is a song entity", 
                             "{} is a plot entity", 
                             "{} is a review entity", 
                             "{} is a character entity", 
                             "{} is a rating entity", 
                             "{} is a ratings average entity", 
                             "{} is a trailer entity", 
                             "{} is not a named entity"]
        elif te == 'te2':
            template_list = ["The entity type of {} is actor",
                             "The entity type of {} is year", 
                             "The entity type of {} is title",
                             "The entity type of {} is genre", 
                             "The entity type of {} is director", 
                             "The entity type of {} is song", 
                             "The entity type of {} is plot", 
                             "The entity type of {} is review", 
                             "The entity type of {} is character", 
                             "The entity type of {} is rating", 
                             "The entity type of {} is ratings average", 
                             "The entity type of {} is trailer", 
                             "The entity type of {} is none entity"]
        elif te == 'te3':
            template_list = ["{} belongs to actor category",
                             "{} belongs to year category", 
                             "{} belongs to title category",
                             "{} belongs to genre category", 
                             "{} belongs to director category", 
                             "{} belongs to song category", 
                             "{} belongs to plot category", 
                             "{} belongs to review category", 
                             "{} belongs to character category", 
                             "{} belongs to rating category", 
                             "{} belongs to ratings average category", 
                             "{} belongs to trailer category", 
                             "{} belongs to none category"]
        elif te == 'te4':
            template_list = ["{} should be tagged as actor",
                             "{} should be tagged as year", 
                             "{} should be tagged as title",
                             "{} should be tagged as genre", 
                             "{} should be tagged as director", 
                             "{} should be tagged as song", 
                             "{} should be tagged as plot", 
                             "{} should be tagged as review", 
                             "{} should be tagged as character", 
                             "{} should be tagged as rating", 
                             "{} should be tagged as ratings average", 
                             "{} should be tagged as trailer", 
                             "{} should be tagged as none entity"]
        else:
            raise NotImplementedError     
    elif dataset_name == 'MIT_MM':
        entity_dict = {0: "actor", 1: "year", 2: "title", 3: "genre", 4: "director", 5: "song", 6: "plot", 7: "review", 8: "character", 9: "rating", 10: "ratings_average", 11: "trailer", 12: "opinion", 13: "award", 14: "origin", 15: "soundtrack", 16: "relationship", 17: "character_name", 18: "quote", 19: "O"}
        if te == 'te1':
            template_list = ["{} is an actor entity",
                             "{} is a year entity", 
                             "{} is a title entity",
                             "{} is a genre entity", 
                             "{} is a director entity", 
                             "{} is a song entity", 
                             "{} is a plot entity", 
                             "{} is a review entity", 
                             "{} is a character entity", 
                             "{} is a rating entity", 
                             "{} is a ratings average entity", 
                             "{} is a trailer entity", 
                             "{} is an opinion entity", 
                             "{} is an award entity",
                             "{} is an origin entity",
                             "{} is a soundtrack entity",
                             "{} is a relationship entity",
                             "{} is a character name entity",
                             "{} is a quote entity",
                             "{} is not a named entity"]
        elif te == 'te2':
            template_list = ["The entity type of {} is actor",
                             "The entity type of {} is year", 
                             "The entity type of {} is title",
                             "The entity type of {} is genre", 
                             "The entity type of {} is director", 
                             "The entity type of {} is song", 
                             "The entity type of {} is plot", 
                             "The entity type of {} is review", 
                             "The entity type of {} is character", 
                             "The entity type of {} is rating", 
                             "The entity type of {} is ratings average", 
                             "The entity type of {} is trailer", 
                             "The entity type of {} is opinion",
                             "The entity type of {} is award",
                             "The entity type of {} is origin",
                             "The entity type of {} is soundtrack",
                             "The entity type of {} is relationship",
                             "The entity type of {} is character name",
                             "The entity type of {} is quote",
                             "The entity type of {} is none entity"]
        elif te == 'te3':
            template_list = ["{} belongs to actor category",
                             "{} belongs to year category", 
                             "{} belongs to title category",
                             "{} belongs to genre category", 
                             "{} belongs to director category", 
                             "{} belongs to song category", 
                             "{} belongs to plot category", 
                             "{} belongs to review category", 
                             "{} belongs to character category", 
                             "{} belongs to rating category", 
                             "{} belongs to ratings average category", 
                             "{} belongs to trailer category", 
                             "{} belongs to opinion category",
                             "{} belongs to award category",
                             "{} belongs to origin category",
                             "{} belongs to soundtrack category",
                             "{} belongs to relationship category",
                             "{} belongs to character name category",
                             "{} belongs to quote category",
                             "{} belongs to none category"]
        elif te == 'te4':
            template_list = ["{} should be tagged as actor",
                             "{} should be tagged as year", 
                             "{} should be tagged as title",
                             "{} should be tagged as genre", 
                             "{} should be tagged as director", 
                             "{} should be tagged as song", 
                             "{} should be tagged as plot", 
                             "{} should be tagged as review", 
                             "{} should be tagged as character", 
                             "{} should be tagged as rating", 
                             "{} should be tagged as ratings average", 
                             "{} should be tagged as trailer", 
                             "{} should be tagged as opinion",
                             "{} should be tagged as award",
                             "{} should be tagged as origin",
                             "{} should be tagged as soundtrack",
                             "{} should be tagged as relationship",
                             "{} should be tagged as character name",
                             "{} should be tagged as quote",
                             "{} should be tagged as none entity"]
        else:
            raise NotImplementedError
    elif dataset_name == 'MIT_R':
        entity_dict = {0: "Rating", 1: "Amenity", 2: "Location", 3: "Restaurant_Name", 4: "Price", 5: "Hours", 6: "Dish", 7: "Cuisine", 8: "O"}
        if te == 'te1':
            template_list = ["{} is a rating entity",
                             "{} is an amenity entity", 
                             "{} is a location entity",
                             "{} is a restaurant name entity", 
                             "{} is a price entity", 
                             "{} is a hours entity", 
                             "{} is a dish entity", 
                             "{} is a cuisine entity", 
                             "{} is not a named entity"]
        elif te == 'te2':
            template_list = ["The entity type of {} is rating",
                             "The entity type of {} is amenity", 
                             "The entity type of {} is location",
                             "The entity type of {} is restaurant name", 
                             "The entity type of {} is price", 
                             "The entity type of {} is hours", 
                             "The entity type of {} is dish", 
                             "The entity type of {} is cuisine", 
                             "The entity type of {} is none entity"]
        elif te == 'te3':
            template_list = ["{} belongs to rating category",
                             "{} belongs to amenity category", 
                             "{} belongs to location category",
                             "{} belongs to restaurant name category", 
                             "{} belongs to price category", 
                             "{} belongs to hours category", 
                             "{} belongs to dish category", 
                             "{} belongs to cuisine category", 
                             "{} belongs to none category"]
        elif te == 'te4':
            template_list = ["{} should be tagged as rating",
                             "{} should be tagged as amenity", 
                             "{} should be tagged as location",
                             "{} should be tagged as restaurant name", 
                             "{} should be tagged as price", 
                             "{} should be tagged as hours", 
                             "{} should be tagged as dish", 
                             "{} should be tagged as cuisine", 
                             "{} should be tagged as none entity"]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return template_list, entity_dict


def read_data(dataset_name, meta_task_num, data_dir_path, K, te, have_dev, train_sample_number, seed, USE_BE):
    if te == 'te1':
        if dataset_name == 'cnoll2003':
            train_file_path = data_dir_path + dataset_name + '/train/train_{}.csv'.format(te)
            eval_file_path = data_dir_path + dataset_name + '/dev/dev_{}.csv'.format(te)
            train_data = pd.read_csv(train_file_path, sep=',').values.tolist()
            train_df = pd.DataFrame(train_data, columns=["input_text", "target_text"])
            if train_sample_number != -1:
                train_df = Sample(train_df, train_sample_number, random_state=seed, te=te)
                # if args.seed == -1:
                #     train_df.to_csv('/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/hanchengcheng02/FewNER-explore/templateNER-main/sample_data/{}.csv'.format(file_dir))
            if have_dev is True:
                eval_data = pd.read_csv(eval_file_path, sep=',').values.tolist()
                eval_df = pd.DataFrame(eval_data, columns=["input_text", "target_text"])
            else:
                eval_df = train_df
        elif dataset_name == 'MIT_M':
            if USE_BE is True:
                train_file_path = os.path.join(data_dir_path, 'be_data', dataset_name, 'mit_m.' + str(K) + '_shot', 'train' + str(meta_task_num) + '.csv')
            else:
                train_file_path = os.path.join(data_dir_path, dataset_name, 'mit_m.' + str(K) + '_shot', 'train' + str(meta_task_num) + '.csv')
            eval_file_path = train_file_path
            train_data = pd.read_csv(train_file_path, sep=',').values.tolist()
            train_df = pd.DataFrame(train_data, columns=["input_text", "target_text", "O-entity"])
            if train_sample_number != -1:
                train_df = Sample(train_df, train_sample_number, random_state=seed, te=te)
                # if args.seed == -1:
                #     train_df.to_csv('/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/hanchengcheng02/FewNER-explore/templateNER-main/sample_data/{}.csv'.format(file_dir))
            if have_dev is True:
                eval_data = pd.read_csv(eval_file_path, sep=',').values.tolist()
                eval_df = pd.DataFrame(eval_data, columns=["input_text", "target_text", "O-entity"])
            else:
                eval_df = train_df
        elif dataset_name == 'MIT_MM':
            if USE_BE is True:
                train_file_path = os.path.join(data_dir_path, 'be_data', dataset_name, 'mit_m.merge.' + str(K) + '_shot', 'train' + str(meta_task_num) + '.csv')
            else:
                train_file_path = os.path.join(data_dir_path, dataset_name, 'mit_m.merge.' + str(K) + '_shot', 'train' + str(meta_task_num) + '.csv')
            eval_file_path = train_file_path
            train_data = pd.read_csv(train_file_path, sep=',').values.tolist()
            train_df = pd.DataFrame(train_data, columns=["input_text", "target_text", "O-entity"])
            if train_sample_number != -1:
                train_df = Sample(train_df, train_sample_number, random_state=seed, te=te)
                # if args.seed == -1:
                #     train_df.to_csv('/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/hanchengcheng02/FewNER-explore/templateNER-main/sample_data/{}.csv'.format(file_dir))
            if have_dev is True:
                eval_data = pd.read_csv(eval_file_path, sep=',').values.tolist()
                eval_df = pd.DataFrame(eval_data, columns=["input_text", "target_text", "O-entity"])
            else:
                eval_df = train_df
        elif dataset_name == 'MIT_R':
            if USE_BE is True:
                train_file_path = os.path.join(data_dir_path, 'be_data', dataset_name, 'mit_r.' + str(K) + '_shot', 'train' + str(meta_task_num) + '.csv')
            else:
                train_file_path = os.path.join(data_dir_path, dataset_name, 'mit_r.' + str(K) + '_shot', 'train' + str(meta_task_num) + '.csv')
            eval_file_path = train_file_path
            train_data = pd.read_csv(train_file_path, sep=',').values.tolist()
            train_df = pd.DataFrame(train_data, columns=["input_text", "target_text", "O-entity"])
            if train_sample_number != -1:
                train_df = Sample(train_df, train_sample_number, random_state=seed, te=te)
                # if args.seed == -1:
                #     train_df.to_csv('/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/hanchengcheng02/FewNER-explore/templateNER-main/sample_data/{}.csv'.format(file_dir))
            if have_dev is True:
                eval_data = pd.read_csv(eval_file_path, sep=',').values.tolist()
                eval_df = pd.DataFrame(eval_data, columns=["input_text", "target_text", "O-entity"])
            else:
                eval_df = train_df 
        else:
            raise NotImplementedError
    elif te in ['te2', 'te3', 'te4']:
        if dataset_name == 'cnoll2003':
            train_file_path = data_dir_path + dataset_name + '/train/train_{}.csv'.format(te)
            eval_file_path = data_dir_path + dataset_name + '/dev/dev_{}.csv'.format(te)
            train_data = pd.read_csv(train_file_path, sep=',').values.tolist()
            train_df = pd.DataFrame(train_data, columns=["input_text", "target_text"])
            if train_sample_number != -1:
                train_df = Sample(train_df, train_sample_number, random_state=seed, te=te)
                # if args.seed == -1:
                #     train_df.to_csv('/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/hanchengcheng02/FewNER-explore/templateNER-main/sample_data/{}.csv'.format(file_dir))
            if have_dev is True:
                eval_data = pd.read_csv(eval_file_path, sep=',').values.tolist()
                eval_df = pd.DataFrame(eval_data, columns=["input_text", "target_text"])
            else:
                eval_df = train_df
        elif dataset_name == 'MIT_M':
            if USE_BE is True:
                train_file_path = os.path.join(data_dir_path, 'te_bedata', te, dataset_name, 'mit_m.' + str(K) + '_shot', 'train' + str(meta_task_num) + '.csv')
            else:
                train_file_path = os.path.join(data_dir_path, 'te_data', te, dataset_name, 'mit_m.' + str(K) + '_shot', 'train' + str(meta_task_num) + '.csv')
            eval_file_path = train_file_path
            train_data = pd.read_csv(train_file_path, sep=',').values.tolist()
            train_df = pd.DataFrame(train_data, columns=["input_text", "target_text", "O-entity"])
            if train_sample_number != -1:
                train_df = Sample(train_df, train_sample_number, random_state=seed, te=te)
                # if args.seed == -1:
                #     train_df.to_csv('/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/hanchengcheng02/FewNER-explore/templateNER-main/sample_data/{}.csv'.format(file_dir))
            if have_dev is True:
                eval_data = pd.read_csv(eval_file_path, sep=',').values.tolist()
                eval_df = pd.DataFrame(eval_data, columns=["input_text", "target_text", "O-entity"])
            else:
                eval_df = train_df
        elif dataset_name == 'MIT_MM':
            if USE_BE is True:
                train_file_path = os.path.join(data_dir_path, 'te_bedata', te, dataset_name, 'mit_m.merge.' + str(K) + '_shot', 'train' + str(meta_task_num) + '.csv')
            else:
                train_file_path = os.path.join(data_dir_path, 'te_data', te, dataset_name, 'mit_m.merge.' + str(K) + '_shot', 'train' + str(meta_task_num) + '.csv')
            eval_file_path = train_file_path
            train_data = pd.read_csv(train_file_path, sep=',').values.tolist()
            train_df = pd.DataFrame(train_data, columns=["input_text", "target_text", "O-entity"])
            if train_sample_number != -1:
                train_df = Sample(train_df, train_sample_number, random_state=seed, te=te)
                # if args.seed == -1:
                #     train_df.to_csv('/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/hanchengcheng02/FewNER-explore/templateNER-main/sample_data/{}.csv'.format(file_dir))
            if have_dev is True:
                eval_data = pd.read_csv(eval_file_path, sep=',').values.tolist()
                eval_df = pd.DataFrame(eval_data, columns=["input_text", "target_text", "O-entity"])
            else:
                eval_df = train_df
        elif dataset_name == 'MIT_R':
            if USE_BE is True:
                train_file_path = os.path.join(data_dir_path, 'te_bedata', te, dataset_name, 'mit_r.' + str(K) + '_shot', 'train' + str(meta_task_num) + '.csv')
            else:
                train_file_path = os.path.join(data_dir_path, 'te_data', te, dataset_name, 'mit_r.' + str(K) + '_shot', 'train' + str(meta_task_num) + '.csv')
            eval_file_path = train_file_path
            train_data = pd.read_csv(train_file_path, sep=',').values.tolist()
            train_df = pd.DataFrame(train_data, columns=["input_text", "target_text", "O-entity"])
            if train_sample_number != -1:
                train_df = Sample(train_df, train_sample_number, random_state=seed, te=te)
                # if args.seed == -1:
                #     train_df.to_csv('/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/hanchengcheng02/FewNER-explore/templateNER-main/sample_data/{}.csv'.format(file_dir))
            if have_dev is True:
                eval_data = pd.read_csv(eval_file_path, sep=',').values.tolist()
                eval_df = pd.DataFrame(eval_data, columns=["input_text", "target_text", "O-entity"])
            else:
                eval_df = train_df 
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError    
    

    return train_df, eval_df


def read_test_data(file_path, part_num=9999999):
    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            words = []
            labels = []
            if line.startswith("Source sentence"):
                continue
            else:
                splits = line.split(",\t")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
            if words:
                examples.append(InputExample(words=words, labels=labels))
            if i >= part_num:
                break
    return examples


def get_CM(train_df, template_list, entity_dict, tokenizer, model, device, args):
    train_examples = []
    for i in range(train_df.shape[0]):
        train_examples.append(Trainingset_InputExample(train_df['input_text'][i], train_df['target_text'][i]))
    model.to(device)
    train_logits = []
    train_label = []
    for train_example in train_examples:
        input_words = train_example.inputs.strip().split()
        output_words = train_example.outputs.strip().split()
        train_logits_lst, label = train_prediction(input_words, output_words, template_list, entity_dict, tokenizer, model, device, args)
        train_logits.append(np.array(train_logits_lst))
        train_label.append(label)
    all_mean = []
    train_logits = np.array(train_logits)
    train_label = np.array(train_label)
    for i in range(len(template_list)):
        temp_mean = np.mean(train_logits[train_label == i], axis=0)
        all_mean.append(temp_mean)
    Calibrate_Metric = np.mean(all_mean, axis=0)

    if args.CM_mode == 'softmax':
        return F.softmax(torch.tensor(Calibrate_Metric.tolist()))  # [0.1106, 0.1104...]
    elif args.CM_mode == '1/softmax':
        return 1 / F.softmax(torch.tensor(Calibrate_Metric.tolist()))  # tensor([9.0379, 9.0613, 9.0255, 9.0268, 9.0380, 9.0751, 9.0077, 9.0122, 8.7256])
    elif args.CM_mode == 'log_softmax':
        return F.softmax(torch.log(torch.tensor(Calibrate_Metric.tolist())))  # 
    elif args.CM_mode == '1/CM':
        return 1 / torch.tensor(Calibrate_Metric.tolist())
    elif args.CM_mode == '1/CM_softmax':
        return F.softmax(torch.tensor((1 / Calibrate_Metric).tolist()))
    elif args.CM_mode == 'layer_norm':
        temp = torch.tensor((Calibrate_Metric).tolist(), dtype=torch.float)
        return F.sigmoid(F.layer_norm(1/temp, normalized_shape=[temp.shape[0]], bias=torch.full((temp.shape[0],), args.ln_bias).float()))