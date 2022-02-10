import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from collections import OrderedDict
from transformers import BertTokenizer
import random
import math
import json
from fewner.sample import create_sample
from fewner.load_xml import get_all_onto_sentence, read_json
import copy

class OntonotesReader:
    def __init__(self, data_path: str, tokenizer: BertTokenizer,ways: int,
                 shots:int,qry_sen_num:int):
        self.file, self.total_type = get_all_onto_sentence(data_path)
        self.tokenizer = tokenizer
        self.ways = ways
        self.shots = shots
        self.qry_sen_num = qry_sen_num
        
    def parse_dataset(self, randomseed=0):
        random.seed(randomseed)
        label = random.sample(self.total_type, self.ways)
        
        spt_dataset = GeniaDataset(label, self.tokenizer)
        qry_dataset = GeniaDataset(label, self.tokenizer)
        
        total = target_sentence(self.file,label)
        spt_sample, qry_sample = parse_N_WAY_K_SHOT(total, label, self.shots,self.qry_sen_num)
        spt_dataset.creat_document(spt_sample)
        qry_dataset.creat_document(qry_sample)
        
        return spt_dataset, qry_dataset
    

class BioNLPReader:
    def __init__(self, data_path: str, tokenizer: BertTokenizer,
                 ways: int, shots:int):
        self.file, self.total_type = read_json(data_path)
        self.tokenizer = tokenizer
        self.ways = ways
        self.shots = shots
        

    def parse_dataset(self,
                      test_sample_num: int,
                      randomseed=0):
        random.seed(randomseed)
        label = random.sample(self.total_type, self.ways)
        total = target_sentence(self.file,label)

        outer_test_dataset = GeniaDataset(label, self.tokenizer)
        outer_test_dataset.dataset_label = 'outer_test_dataset'

        inner_test_dataset = GeniaDataset(label, self.tokenizer)
        inner_test_dataset.dataset_label = 'inner_test_dataset'
        
        spt_sample, qry_sample = parse_N_WAY_K_SHOT(total, label, self.shots,test_sample_num)

        outer_test_dataset.creat_document(spt_sample)
        outer_test_dataset.dataset_label = 'train'

        inner_test_dataset.creat_document(qry_sample)
        inner_test_dataset.dataset_label = 'test'
        return outer_test_dataset, inner_test_dataset

class GeniaDataset(TorchDataset):
    def __init__(self, label: list, tokenizer: BertTokenizer):
        super(GeniaDataset, self).__init__()
        self.dataset_label = ''
        self.sentences = OrderedDict()
        self.labels = OrderedDict()
        self.spans = OrderedDict()
        self.token_idx = OrderedDict()
        self.label_vocab = self._creat_vocab(label)
        self._doc_id = 0
        self._tokenizer = tokenizer

    def _creat_vocab(self, label: list):
        label_vocab = {}
        for i in range(len(label)):
            label_vocab[label[i]] = i + 1
        return label_vocab

    def creat_document(self, data: list):
        for i in data:
            doc_encoding, label, span, token_idx = token_sentence(i, self._tokenizer)
            self.sentences[self._doc_id] = doc_encoding
            self.labels[self._doc_id] = [self.label_vocab[i] for i in label]
            self.spans[self._doc_id] = span
            self.token_idx[self._doc_id] = token_idx
            self._doc_id += 1


    def creat_classify_document(self, doc_encoding, label, span, token_idx):
        self.sentences[self._doc_id] = doc_encoding
        self.labels[self._doc_id] = label
        self.spans[self._doc_id] = span
        self.token_idx[self._doc_id] = token_idx
        self._doc_id += 1

    def static_label_num(self):
        result = {}
        for i in self.labels.values():
            for j in i:
                if j not in result:
                    result[j] = 1
                else:
                    result[j] += 1
        return result

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        doc = dict(
            sentence=self.sentences[index],
            label=self.labels[index],
            span=self.spans[index],
            token_idx=self.token_idx[index]
        )
        if self.dataset_label == 'train':
            result = create_sample(doc)
        else:
            result = create_sample(doc, neg_sample=False)
        return result

def target_sentence(file:list, label:list):
    total = []
    for i in file:
        current = copy.deepcopy(i)
        for j in range(len(current['type'])-1,-1,-1):
            if current['type'][j]['type'] not in label:
                current['type'].pop(j)
        if current['type'] != []:
            total.append(current)
    return total
    
def needsample(inner_label_num: dict, shots: int):
    for i in list(inner_label_num.values()):
        if i < shots:
            return True
    return False
  
def label_need_sample(inner_label_num: dict, shots: int):
    result = []
    for key, value in inner_label_num.items():
        if value < shots:
            result.append(key)
    return result
  
def parse_N_WAY_K_SHOT(file:list, label:list, shots:int, qry_sen_num:int):
    
    spt_sample, qry_sample = [], []
    spt_label_num = {}
    for i in label:
        spt_label_num[i] = 0
    while needsample(spt_label_num, shots):
        type_need_sample = label_need_sample(spt_label_num, shots)
        sample = random.choice(file)
        types = [i['type'] for i in sample['type']]
        retain = False
        for i in types:
            if i in type_need_sample:
                retain = True
                break
        if retain == True:
            spt_sample.append(sample)
            for i in types:
                spt_label_num[i] += 1
            file.remove(sample)
    qry_sample = random.sample(file, min(len(file),qry_sen_num))
    
    return spt_sample, qry_sample
    

def token_sentence(data: dict, tokenizer: BertTokenizer):
    token_idx = []
    doc_encoding = [tokenizer.convert_tokens_to_ids('[CLS]')]
    idx = 1
    for i in data['sentence']:
        token_encoding = tokenizer.encode(i, add_special_tokens=False)
        token_idx.append(idx)
        idx += len(token_encoding)
        doc_encoding += token_encoding
    doc_encoding += [tokenizer.convert_tokens_to_ids('[SEP]')]
    token_idx.append(idx)
    label = []
    span = []
    for i in data['type']:
        label.append(i['type'])
        start = i['start']
        end = i['end']
        start = token_idx[start]
        end = token_idx[end]
        span.append([start, end])
    return doc_encoding, label, span, token_idx

if __name__ == '__main__':
    json_path = r'data/ontonotes'
    # test =
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased',
                                              do_lower_case=False)
    
    a = OntonotesReader(json_path, tokenizer,ways=5,shots=5,qry_sen_num=15)
    spt_dataset, qry_dataset = a.parse_dataset()

    json_path = r'data/BioNLP'
    a = BioNLPReader(json_path, tokenizer,ways=5,shots=5)
    outer_test_dataset, inner_test_dataset = a.parse_dataset(test_sample_num=5000)

    print()