import csv
from transformers import AlbertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import sys, os
from datetime import datetime
import pandas as pd
import jieba
import numpy as np

all_label = ['BACKGROUND', 'OBJECTIVES', 'METHODS', 'RESULTS', 'CONCLUSIONS', 'OTHERS']

def log(*logs):
    enablePrint()
    print(*logs)
    blockPrint()

def computeAccuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100

def saveModel(model,name):
    now = datetime.now()
    base_dir = 'train_models/'
    save_dir = base_dir + now.strftime("%m-%d-%Y_%H-%M-%S_") + name
    os.mkdir(save_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(save_dir)

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def makeTorchDataLoader(torch_dataset,batch_size = 16):
    return DataLoader(torch_dataset,batch_size=batch_size,shuffle=True)

def makeTorchDataSet(mr_data_class,is_train_data = True):
    all_input_phrase_ids = []
    all_input_ids = []
    all_token_type_ids = []
    all_position_ids = []
    all_answer_lables = []
    max_input_length = 512
    for d in mr_data_class.data:
        sentenceId = d[0]
        context_ids = d[1]
        sentence_ids = d[2]
        input_ids = d[3]
        label = d[4]

        all_input_phrase_ids.append(sentenceId)
        all_input_ids.append(input_ids)

        token_type_ids = [0]*len(context_ids) + [1]*len(sentence_ids)
        while(len(token_type_ids)<max_input_length):
            token_type_ids.append(0)
        assert len(token_type_ids) == max_input_length
        all_token_type_ids.append(token_type_ids)

        position_ids = [1]*len(input_ids)
        while(len(position_ids)<max_input_length):
            position_ids.append(0)
        assert len(position_ids) == max_input_length
        all_position_ids.append(position_ids)

        if(is_train_data):
            Sentiment = label
            all_answer_lables.append(int(Sentiment))
    
    for input_ids in all_input_ids:
        while(len(input_ids)<max_input_length):
            input_ids.append(0)
        assert len(input_ids) == max_input_length
    
    print(np.array(all_input_ids).shape)
    print(np.array(all_token_type_ids).shape)
    print(np.array(all_position_ids).shape)
    print(np.array(all_answer_lables).shape)

    if(is_train_data):
        torch_input_ids = torch.tensor([ids for ids in all_input_ids], dtype=torch.long)
        torch_token_type_ids = torch.tensor([ids for ids in all_token_type_ids], dtype=torch.long)
        torch_position_ids = torch.tensor([ids for ids in all_position_ids], dtype=torch.long)
        torch_answer_lables = torch.tensor([answer_lable for answer_lable in all_answer_lables], dtype=torch.long)
        return TensorDataset(torch_input_ids, torch_token_type_ids, torch_position_ids, torch_answer_lables)
    else:
        torch_input_ids = torch.tensor([ids for ids in all_input_ids], dtype=torch.long)
        torch_phrase_ids = torch.tensor([int(ids) for ids in all_input_phrase_ids], dtype=torch.long)
        return TensorDataset(torch_input_ids,torch_phrase_ids)

def splitDataset(full_dataset,split_rate = 0.8):
    train_size = int(split_rate * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    return train_dataset, test_dataset

def cutSantenceWithJieba(santence):
    words = jieba.cut(santence, cut_all=False)
    result = ''
    for word in words:
        if word != ' ':
            result += word + ' '
    return result[:-1]

class MR_Data:
    def __init__(self, is_train_data, data):
        self.data = data
        self.is_train_data = is_train_data
    
    @classmethod
    def load_data(cls, path, is_train_data = True):
        tokenizer = AlbertTokenizer.from_pretrained('model/albert-large-spiece.model')
        def toBertIds(context, sentence):
            a_sentence = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(context))
            b_sentence = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence))
            return tokenizer.build_inputs_with_special_tokens(a_sentence, b_sentence)

        def singleBertIds(context):
            a_sentence = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(context))
            return tokenizer.build_inputs_with_special_tokens(a_sentence)

        data_set = pd.read_csv(path)
        new_rows = []
        sentenceId = 0
        for i in range(0, 5):
            paper_title = data_set['Id'][i]
            paper_content = data_set['Abstract'][i]
            paper_content_list = paper_content.split('$$$')
            context = ''
            for sentence in paper_content_list:
                context += cutSantenceWithJieba(sentence) + ' '
            context = context[:-1]

            if is_train_data:
                paper_labels = data_set['Task 1'][i]
                paper_labels = paper_labels.split(' ')
                for index, sentence in enumerate(paper_content_list):
                    paper_label = paper_labels[index].split(' ')
                    for each_paper_label in paper_label:
                        for single_labels in each_paper_label.split('/'):
                            new_row = []
                            idput_ids = toBertIds(context, sentence)
                            if len(idput_ids) <= 512:
                                context_ids = singleBertIds(context)
                                sentence_ids = singleBertIds(sentence)
                                sentence_ids = sentence_ids[1:]
                                new_row = [sentenceId, context_ids, sentence_ids, idput_ids, all_label.index(single_labels)]
                                new_rows.append(new_row)
                                sentenceId += 1
            if i % 1000 == 0: print('load_data:', i)
        return cls(is_train_data, new_rows)
    
    @property
    def total_topic(self):
        return len(self.data) 
        
if __name__ == "__main__":
    pass
    # print(torch.cuda.is_available())
    # is_train_data = True
    # FullData = MR_Data.load_data('dataset/task1_trainset.csv', is_train_data)
    # FullDataset = makeTorchDataSet(FullData, is_train_data)

    # TestData = MR_Data.load_data('dataset/test.tsv')
    # print('TestData.total_topic:',TestData.total_topic)