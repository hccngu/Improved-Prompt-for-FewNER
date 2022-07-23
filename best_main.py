# import string
import pandas as pd
import time
import os
import numpy as np
import logging
import argparse
from utils_metrics import get_entities_bio, f1_score, classification_report
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
from BartModel import BartForConditionalGenerationDoubleDecoder
import torch
import torch.nn as nn
import time
import math
import random
import torch.nn.functional as F
from utils import *
from test import test
from best_model import Seq2SeqModel
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
from transformers import logging as lg
lg.set_verbosity_error()
# from inference import InputExample, prediction, cal_time
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
import warnings
warnings.filterwarnings("ignore")  # 忽略警告信息


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        # self.args = args
        self.fc = nn.Linear(input_dim, output_dim)
        self.loss_func_MLP = nn.CrossEntropyLoss()
    
    def forward(self, data):
        return F.softmax(self.fc(data))


def load_model(ckpt):
    '''
    ckpt: Path of the checkpoint
    return: Checkpoint dict
    '''
    if os.path.isfile(ckpt):
        checkpoint = torch.load(ckpt)
        print("Successfully loaded checkpoint '%s'" % ckpt)
        return checkpoint
    else:
        raise Exception("No checkpoint found at '%s'" % ckpt)


def add_labels_row(df):
    labels = []
    for true in df['true_list']:
        true = eval(true)
        if len(true) == 1:
            if true[0] == 'O':
                labels.append('O')
            else:
                labels.append(true[0][2:])
        else:
            if true[0] == 'O':
                labels.append('O')
            else:
                flag = False
                temp = true[0][2:]
                for l in true:
                    if l == 'O':
                        flag = True
                        break
                    else:
                        if l[2:] != temp:
                            flag = True
                            break
                if flag is True:
                    labels.append('O')
                else:
                    labels.append(temp)
    df['labels'] = labels


def add_logits_rows(df):
    lll1, lll2, lll3, lll4, lll5 = [], [], [], [], []
    for i, ll in enumerate(df['logits_list']):
        lll = eval(ll)
        lll1.append(lll[0])
        lll2.append(lll[1])
        lll3.append(lll[2])
        lll4.append(lll[3])
        lll5.append(lll[4])
        df['logits_list'][i] = ll
    df['LOC_logits'] = lll1
    df['PER_logits'] = lll2
    df['ORG_logits'] = lll3
    df['MISC_logits'] = lll4
    df['O_logits'] = lll5


def get_draw_data(df):
    LOC_data = df[df['labels'].isin(['LOC'])]
    all_data = LOC_data[['LOC_logits', 'PER_logits', 'ORG_logits', 'MISC_logits', 'O_logits']].values.T
    return all_data


def draw(df, save_path):
    labels   = ['LOC', 'PER', 'ORG', 'MISC', 'O']

    # 1行 2列 ，长9英寸 宽4英寸
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 9))

    for i, label in enumerate(labels):
        _data = df[df['labels'].isin([label])]
        all_data = _data[['LOC_logits', 'PER_logits', 'ORG_logits', 'MISC_logits', 'O_logits']].values.T
        all_data2 = []
        for j in range(all_data.shape[0]):
            all_data2.append(all_data[j])

        bplot = axes[i].boxplot(all_data2,
                            vert=True,  #是否需要将箱线图垂直摆放，默认垂直摆放；
                            patch_artist=True,   # 是否填充箱体的颜色；
                            labels=labels,
                            meanline=True,
                            showmeans=True,
                            whis=0.75)   # 为箱线图添加标签，类似于图例的作用；
        axes[i].set_title('{}_logits'.format(label))

        colors = ['pink', 'lightblue', 'lightgreen', 'orange', 'violet']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

        axes[i].yaxis.grid(True)
        axes[i].set_xlabel('labels')
        axes[i].set_ylabel('logits')

    
    plt.show()
    plt.savefig(save_path)


def template_entity(words, input_TXT, start, template_list, tokenizer, model, device, mlp, args):
    # input text -> template
    words_length = len(words)
    # words_length_list = [len(i) for i in words]
    input_TXT = [input_TXT]*(5*words_length)

    input_ids = tokenizer(input_TXT, return_tensors='pt')['input_ids']
    model.to(device)
    mlp.to(device)
    # template_list = [" is a location entity .", " is a person entity .", " is an organization entity .",
    #                  " is an other entity .", " is not a named entity ."]
    entity_dict = {0: 'LOC', 1: 'PER', 2: 'ORG', 3: 'MISC', 4: 'O'}
    temp_list = []
    for i in range(words_length):
        for j in range(len(template_list)):
            temp_list.append(template_list[j].format(words[i]))

    output_ids = tokenizer(temp_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
    output_ids[:, 0] = 2  # why
    output_length_list = [0]*5*words_length


    for i in range(len(temp_list)//5):
        base_length = ((tokenizer(temp_list[i * 5], return_tensors='pt', padding=True, truncation=True)['input_ids']).shape)[1] - 4
        output_length_list[i*5:i*5+ 5] = [base_length]*5
        output_length_list[i*5+4] += 1

    score = [1]*5*words_length
    with torch.no_grad():
        origin_output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids[:, :output_ids.shape[1] - 2].to(device), output_hidden_states=True)
        output = origin_output[0]
        for i in range(output_ids.shape[1] - 3):
            # print(input_ids.shape)
            logits = output[:, i, :]
            logits = logits.softmax(dim=1)
            # values, predictions = logits.topk(1,dim = 1)
            logits = logits.to('cpu').numpy()
            # print(output_ids[:, i+1].item())
            for j in range(0, 5*words_length):
                if i < output_length_list[j]:
                    score[j] = score[j] * logits[j][int(output_ids[j][i + 1])]

        # add mlp
        if args.use_mlp_or_not is 'yes':
            input_mlp = origin_output.decoder_hidden_states[-1][0, (len(words[0].split())-1), :]
            mlp_logits = mlp(input_mlp)
            # without_O_mlp_logits = 0
            # O_mlp_logit = 0
            norm_score = (score-min(score))/(max(score)-min(score))
            norm_score = F.softmax(torch.tensor(norm_score)).numpy()
            if np.argmax(norm_score) != (norm_score.shape[0]-1):
                norm_score[:-1] = norm_score[:-1] * mlp_logits[1].item()
                norm_score[-1] *= mlp_logits[0].item()


    end = start + (len(words[0].split())-1)
        # score_list.append(score)
    norm_score = norm_score.tolist()
    # s_lst = norm_score[(norm_score.index(max(norm_score))//5)*5:(norm_score.index(max(norm_score))//5)*5+5]
    # norm_s_lst = (s_lst-min(s_lst))/(max(s_lst)-min(s_lst))
    return [start, end, entity_dict[(norm_score.index(max(norm_score))%5)], max(norm_score), norm_score, mlp_logits.cpu().numpy().tolist()]  #[start_index,end_index,label,score,[score_list],[mlp_logits]]


def prediction(input_TXT, template_list, tokenizer, model, device, mlp, args):
    input_TXT_list = input_TXT.split(' ')

    entity_list = []
    for i in range(len(input_TXT_list)):
        words = []
        for j in range(1, min(5, len(input_TXT_list) - i + 1)):
            word = (' ').join(input_TXT_list[i:i+j])
            words.append([word])

        for w in words:
            entity = template_entity(w, input_TXT, i, template_list, tokenizer, model, device, mlp, args) #[start_index,end_index,label,score, score_list]
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
        temp.append(entity[5])
        pred_logits.append(temp)

    
    return label_list, pred_logits


def cal_time(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


class InputExample():
    def __init__(self, words, labels):
        self.words = words
        self.labels = labels


def main():
    
    # Config
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_sample_number', default=-1, type=int, help='just use for cnoll2003')
    parser.add_argument('--te', type=str, default='te1', help='templates, [te1, te2, te3, te4], deflaut=te1')
    parser.add_argument('--cuda', default=0, type=int, help='cuda device.')
    parser.add_argument('--epochs', default=4, type=int, help='the number of epochs. default=20')
    parser.add_argument('--to_draw', default=False, type=bool, help='draw or not')
    parser.add_argument('--seed', default=4, type=int, help='random seed, default=4')
    parser.add_argument('--use_mlp_or_not', default=False, help='use mlp or not')
    parser.add_argument('--info', default='new_start_best_model', help='file_name_prefix')
    parser.add_argument('--K', default=10, type=int, help='K-shot')
    
    # read_data
    parser.add_argument('--dataset', default='MIT_R', help='[cnoll2003, MIT_M, MIT_MM, MIT_R, snips]')
    parser.add_argument('--data_dir', default='/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/hanchengcheng02/FewNER-explore/templateNER-main/data/')
    parser.add_argument('--have_dev', default=True, help='have dev set or not')
    parser.add_argument('--meta_task_num', default=0, type=int, help='meta task number')

    # model config
    # parser.add_argument('--encoder_decoder_type', default='gpt2', help='model_classes [bart, gpt2]')
    # parser.add_argument('--encoder_decoder_name', default='/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/hanchengcheng02/FewNER-explore/templateNER-main/gpt2', help='model_path [bart, gpt2]')
    parser.add_argument('--encoder_decoder_type', default='bart', help='model_classes [bart, bart2decoder, gpt2]')
    parser.add_argument('--encoder_decoder_name', default='/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/hanchengcheng02/FewNER-explore/templatener/bart-large', help='model_path [bart, gpt2]')
    parser.add_argument('--best_model_path', default='/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/hanchengcheng02/FewNER-explore/templateNER-main/outputs/new_start_best_model_mit_r-MIT_R--1-te1-seed-4-1658095533733', help='pretrained best model path')
    parser.add_argument('--do_train', default=False, action='store_true', help='train or not')
    parser.add_argument('--use_cm_or_not', default=False, action='store_true', help='use Calibrate Metric or not')
    parser.add_argument('--CM_mode', default='layer_norm', help='CM_mode, [softmax, 1/softmax, log_softmax, 1/CM, 1/CM_softmax, layer_bias]')
    parser.add_argument('--ln_bias', default=1, type=int, help='When CM_mode is layer norm, it can control layer norm bias.')
    parser.add_argument('--max_entity_length', default=5, type=int, help='max entity length')
    parser.add_argument('--lr', default=4e-5, type=float, help='learning rate, default=4e-5')
    # parser.add_argument('--use_siamese_decoder', default=True, action='store_true', help='use siamese decoder or not')
    args = parser.parse_args()

    # if args.do_train == True:
    #     print('Training...')

    # set_value
    DATASET = args.dataset
    DATA_DIR = args.data_dir
    TEMPLATE = args.te
    HAVE_DEV = args.have_dev
    SAVE_PREFIX = args.info
    TRAIN_SAMPLE_NUMBER = args.train_sample_number
    SEED = args.seed
    CUDA = args.cuda
    EDT = args.encoder_decoder_type
    EDN = args.encoder_decoder_name
    EPOCH = args.epochs
    USE_MLP = args.use_mlp_or_not
    DO_TRAIN = args.do_train
    BEST_MODEL_PATH = args.best_model_path
    META_TASK_NUM = args.meta_task_num
    K = args.K
    USE_CM = args.use_cm_or_not
    LR = args.lr
    # USE_SD = args.use_siamese_decoder

    template_list, entity_dict = set_template(DATASET, TEMPLATE)
    
    
    if DO_TRAIN == True:
        # read_data
        train_df, eval_df = read_data(DATASET, META_TASK_NUM, DATA_DIR, K, TEMPLATE, HAVE_DEV, TRAIN_SAMPLE_NUMBER, SEED)

        # set save_model_path
        file_dir = '-'.join([SAVE_PREFIX, EDT, DATASET, str(TRAIN_SAMPLE_NUMBER), TEMPLATE, 'seed', str(SEED), str(META_TASK_NUM), '-shot', str(int(round(time.time() * 1000)))])
        best_model_path = "/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/hanchengcheng02/FewNER-explore/templateNER-main/outputs/{}".format(file_dir)
        
        # model config
        model_args = {
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
            "max_seq_length": 50,
            "train_batch_size": 50,  # 100,
            "num_train_epochs": EPOCH,
            "save_eval_checkpoints": False,
            "save_model_every_epoch": False,
            "evaluate_during_training": True,
            "evaluate_generated_text": True,
            "evaluate_during_training_verbose": True,
            "use_multiprocessing": False,
            "max_length": 25,
            "manual_seed": SEED,
            "save_steps": 11898,
            "gradient_accumulation_steps": 1,
            "output_dir": "./exp/template",
            "best_model_dir": best_model_path,
            "learning_rate_MLP": 1e-3,
            "learning_rate": LR,  # 4e-5
            "use_mlp": USE_MLP,
            # "use_sd": USE_SD,
        }

        # Initialize model
        model = Seq2SeqModel(
            encoder_decoder_type=EDT,
            encoder_decoder_name=EDN,
            args=model_args,
            # use_cuda=False,
            cuda_device=CUDA,
        )


        # Train the model
        model.train_model(train_df, eval_data=eval_df)

        # Evaluate the model
        results = model.eval_model(eval_df)

        # Use the model for prediction
        if EDT != 'bart2decoder':
            if DATASET == 'cnoll2003':
                print(model.predict(["Japan began the defence of their Asian Cup title with a lucky 2-1 win against Syria in a Group C championship match on Friday."]))
                print('Ground truth: Japan is a Location entity.')
            elif DATASET == 'MIT_M':
                print(model.predict(["are there any good romantic comedies out right now ?"]))
                print('Ground truth: romantic comedies is a GENRE entity and right now is an YEAR entity.')
            elif DATASET == 'MIT_MM':
                print(model.predict(["what is a bronx tale rated ?"]))
                print('Ground truth: bronx tale is a title entity and rated is a rating entity.')
            elif DATASET == 'MIT_R':
                print(model.predict(["does this chinese restaurant have private rooms ?"]))
                print('Ground truth: chinese is a Cuisine entity and private rooms is an Amentity entity.')

        print("Training Finish and Testing Start...")
    else:
        best_model_path=BEST_MODEL_PATH
    
    print(args)
    
    if EDT == 'bart':
        tokenizer = BartTokenizer.from_pretrained(EDN)
        # input_TXT = "Japan began the defence of their Asian Cup title with a lucky 2-1 win against Syria in a Group C championship match on Friday ."
        model = BartForConditionalGeneration.from_pretrained(best_model_path)
    elif EDT == 'bart2decoder':
        tokenizer = BartTokenizer.from_pretrained(EDN)
        # input_TXT = "Japan began the defence of their Asian Cup title with a lucky 2-1 win against Syria in a Group C championship match on Friday ."
        model = BartForConditionalGenerationDoubleDecoder.from_pretrained(best_model_path)
    else:
        raise NotImplementedError

    if USE_MLP == True:
        mlp = MLP(1024, 2)
        ckpt = best_model_path + '/MLP.ckpt'
        state_dict = load_model(ckpt)['state_dict']
        state_dict["fc.weight"] = state_dict.pop("weight")
        state_dict["fc.bias"] = state_dict.pop("bias")
        own_state = mlp.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            own_state[name].copy_(param)
        mlp.eval()

    model.eval()
    model.config.use_cache = False
    # input_ids = tokenizer(input_TXT, return_tensors='pt')['input_ids']
    # print(input_ids)
    device = torch.device("cuda:{}".format(CUDA) if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    if USE_CM == True:
        train_df, eval_df = read_data(DATASET, META_TASK_NUM, DATA_DIR, K, TEMPLATE, HAVE_DEV, TRAIN_SAMPLE_NUMBER, SEED)
        Calibrate_Metric = get_CM(train_df, template_list, entity_dict, tokenizer, model, device, args)
        print("Calibrate_Metric: ", Calibrate_Metric)

    if DATASET == 'cnoll2003':
        test_file_path = DATA_DIR + DATASET +'/test/test.txt'
    # elif DATASET == 'mit_m.10_shot':
    #     test_file_path = DATA_DIR + DATASET +'/test' + str(META_TASK_NUM) + '.txt'
    elif DATASET in ['MIT_M', 'MIT_MM', 'MIT_R']:
        test_file_path = os.path.join(DATA_DIR, DATASET, 'test.txt')
    else:
        raise NotImplementedError
    test_examples = read_test_data(test_file_path)
    if USE_MLP == True:
        logits_list = test(test_examples, template_list, entity_dict, tokenizer, model, device, args, mlp)
    elif USE_CM == True:
        logits_list = test(test_examples, template_list, entity_dict, tokenizer, model, device, args, CM=Calibrate_Metric)
    else:
        logits_list = test(test_examples, template_list, entity_dict, tokenizer, model, device, args)

    # data_path = '/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/hanchengcheng02/FewNER-explore/templateNER-main/pic_data/{}.csv'.format(file_dir)
    # data_df = pd.DataFrame(logits_list, columns=['words_list', 'preds', 'max_origin_logits', 'logits_list', 'mlp_logits', 'true_list'])
    # data_df.to_csv(data_path)

    # if args.to_draw is True:
    #     df = pd.read_csv(data_path, index_col=0)
    #     add_labels_row(df)
    #     add_logits_rows(df)
    #     # draw_data = get_draw_data(df)
    #     save_path = '/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/hanchengcheng02/FewNER-explore/templateNER-main/pic/{}.png'.format(file_dir)
    #     draw(df, save_path)


    # for num_point in range(len(preds_list)):
    #     preds_list[num_point] = ' '.join(preds_list[num_point]) + '\n'
    #     trues_list[num_point] = ' '.join(trues_list[num_point]) + '\n'
    # with open('./pred.txt', 'w') as f0:
    #     f0.writelines(preds_list)
    # with open('./gold.txt', 'w') as f0:
    #     f0.writelines(trues_list)

if __name__ == '__main__':
    main()