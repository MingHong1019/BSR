# import os
# import warnings
# from typing import List, Tuple, Dict

import torch

from sklearn.metrics import precision_recall_fscore_support as prfs
# from transformers import BertTokenizer
from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import recall_score
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
import numpy as np

class Evaluator:
    def __init__(self, label_vocab):
        self.pre_result = []
        self.real_result = []
        self.label_vocab = label_vocab
        self.label_vocab_t = self._t_dict(label_vocab)
        # self.num = 0

    def convert_result_to_label(self, pre_result: torch.tensor):
        pre = pre_result.squeeze(0).argmax(dim=-1)

        pre = pre.cpu().numpy().tolist()
        pre = pre if isinstance(pre, list) else [pre]
        pre = [self.label_vocab_t[i] for i in pre]
        for i in pre:
            self.pre_result.append(i)
        # self.num +=1

    def append_real_result(self, real_result: torch.tensor):
        label = real_result.squeeze(0)
        label = label.cpu().numpy().tolist()
        label = [self.label_vocab_t[i] for i in label]
        for i in label:
            self.real_result.append(i)

    def score(self, print_results: bool = False):
        labels = [i for i in self.label_vocab.keys()]
        # labels.remove('O')
        per_type = prfs(self.real_result, self.pre_result, labels=labels, average=None, zero_division=0)
        micro = prfs(self.real_result, self.pre_result, labels=labels, average='micro', zero_division=0)[:-1]
        macro = prfs(self.real_result, self.pre_result, labels=labels, average='macro', zero_division=0)[:-1]

        micro = [m for m in micro]
        macro = [m for m in macro]
        total_support = sum(per_type[-1])
        if print_results:
            self._print_results(per_type, list(micro) + [total_support], list(macro) + [total_support], labels)
        return micro, macro

    def _print_results(self, per_type, micro, macro, labels):
        print()
        columns = ('type', 'precision', 'recall', 'f1-score', 'support')
        row_fmt = "%20s" + (" %12s" * (len(columns) - 1))
        results = [row_fmt % columns, '\n']
        metrics_per_type = []
        results.append('\n')
        for i in range(len(labels)):
            metrics = []
            for j in range(len(per_type)):
                metrics.append(per_type[j][i])
            metrics_per_type.append(metrics)

        for m, t in zip(metrics_per_type, labels):
            results.append(row_fmt % self._get_row(m, t))
            results.append('\n')

        results.append('\n')

        # micro
        results.append(row_fmt % self._get_row(micro, 'micro'))
        results.append('\n')

        # macro
        results.append(row_fmt % self._get_row(macro, 'macro'))

        results_str = ''.join(results)
        print(results_str)

    def _get_row(self, data, label):
        row = [label]
        for i in range(len(data) - 1):
            row.append("%.2f" % (data[i] * 100))
        row.append(data[3])
        return tuple(row)

    def _t_dict(self, dic: dict):
        result = {0:'O'}
        for value, key in enumerate(dic):
            result[value+1] = key
        return result

class Evaluator2:
    def __init__(self, label_vocab):
        self.pre_result = []
        self.real_result = []
        self.label_vocab = label_vocab
        self.label_vocab_t = self._t_dict(label_vocab)

    def convert_result_to_label(self, batch: dict, pre_result: torch.tensor):
        label_score, label_index = pre_result.max(dim=-1)
        label_score = label_score.view(-1).cpu().detach().numpy().tolist()
        label_index = label_index.view(-1).cpu().detach().numpy().tolist()

        token_idx = batch['token_idx'].squeeze(0).cpu().numpy().tolist()[:]
        token_idx.insert(0, 0)
        # token_idx.append(batch['encodings'].size(-1))
        real_span_type = batch['entity_types'].squeeze(0).cpu().numpy().tolist()
        pre = ['O' for i in token_idx]
        real = ['O' for i in token_idx]
        entity_mask = batch['entity_masks'].int().squeeze(0).cpu().numpy().tolist()
        for i in range(len(entity_mask)):
            start = 1
            end = len(entity_mask)
            for j in range(len(entity_mask[i])):
                if entity_mask[i][j] == 1:
                    start = j
                    break
            for j in range(len(entity_mask[i])-1, -1, -1):
                if entity_mask[i][j] == 1:
                    end = j+1
                    break
            start = token_idx.index(start)
            try:
                end = token_idx.index(end)
            except:
                print()

            entity_mask[i] = [start, end]
        pre_mask =entity_mask[:]
        for i in range(len(label_index)-1,-1,-1):
            if label_index[i] == 0:
                label_index.remove(label_index[i])
                label_score.remove(label_score[i])
                pre_mask.remove(pre_mask[i])

        for i in range(len(real_span_type)-1,-1,-1):
            if real_span_type[i] == 0:
                real_span_type.remove(real_span_type[i])
                entity_mask.remove(entity_mask[i])

        self._convert_pre(pre_mask, label_score, label_index, pre)
        self._convert_real(entity_mask, real_span_type, real)

    def _convert_pre(self, entity_mask, label_score, label_index, pre):
        def overlapping(span:list, list:list):
            mark = False
            for i in range(span[0],span[1]):
                if list[i] != 'O':
                    mark = True
            return mark
        if label_score != []:
            z = zip(label_score, label_index, entity_mask)
            label_score, label_index, entity_mask = zip(*sorted(z))
        for i in range(len(entity_mask)):
            if overlapping(entity_mask[i], pre):
                continue
            else:
                for j in range(entity_mask[i][0], entity_mask[i][1]):
                    label = 'B-' if j == entity_mask[i][0] else 'I-'
                    label += self.label_vocab_t[label_index[i]]
                    pre[j] = label
        self.pre_result.append(pre)

    def _convert_real(self, entity_mask, real_span_type, real):
        for i in range(len(entity_mask)):
            for j in range(entity_mask[i][0], entity_mask[i][1]):
                label = 'B-' if j == entity_mask[i][0] else 'I-'
                label += self.label_vocab_t[real_span_type[i]]
                real[j] = label
        self.real_result.append(real)

    def score(self, print_score:bool):
        p = precision_score(self.real_result, self.pre_result, scheme=IOB2)
        r = recall_score(self.real_result, self.pre_result, scheme=IOB2)
        f1 = f1_score(self.real_result, self.pre_result, scheme=IOB2)
        if print_score:
            print("classification report: ")
            print(classification_report(self.real_result, self.pre_result, digits=4, scheme=IOB2))
        return f1


    def _t_dict(self, dic: dict):
        result = {0:'O'}
        for value, key in enumerate(dic):
            result[value+1] = key
        return result





