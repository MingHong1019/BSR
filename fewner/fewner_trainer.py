import argparse
import torch
import torch.nn as nn
from transformers import BertConfig, AdamW
from transformers import BertModel, BertTokenizer
from torch.nn import init
from fewner.sample import collect_fn, create_sample

from fewner.models import Bertmodel
from fewner.evaluator import Evaluator, Evaluator2
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score,precision_score,recall_score



class FewNERTrainer:
    def __init__(self, args: argparse.Namespace, x):
        super(FewNERTrainer, self).__init__()
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path,
                                                      do_lower_case=False)
        self.x =x
        self.device_ids = [0]
        # self.device = torch.device('cuda:0')
        self.bertmodel = self._load_model().cuda(device=self.device_ids[0])
        # self.bertmodel = torch.nn.DataParallel(self.bertmodel, device_ids=self.device_ids).cuda(device=self.device_ids[0])
        self.calssify_loss = nn.CrossEntropyLoss(reduction='none')
        self.calssify_optm = AdamW(self._get_optimizer_params(self.bertmodel), lr=args.calssify_lr, correct_bias=False)
        self.test_optm = AdamW(self._get_optimizer_params(self.bertmodel), lr=args.calssify_lr, correct_bias=False)
        self.reduct_optm = torch.optim.Adam(self.bertmodel.parameters(), lr=args.reduction_lr)

        self.softmax = nn.Softmax(dim=-1)
        self.norm = nn.LayerNorm(args.ways, elementwise_affine=False)

        # self._init_model(self.reduct_model)

    def train(self,data_reader, outer_test_dataset, inner_test_dataset):
        # 降维时不计算O，使用线性层做分类
        #outer_train_dataset = data_reader.parse_dataset(run_args.ways, run_args.outer_sample_num, seed)
        for i in range(3):
            classify_train_dataset, classify_dev_dataset = data_reader.parse_dataset()
            
            self.outer_classify_train(classify_train_dataset, classify_dev_dataset, self.args.calssify_train_epoch)

        self.outer_classify_test(outer_test_dataset, trainepoch=30)

        x = self.inner_classify_test(inner_test_dataset)
        

        return x

    def outer_classify_train(self,classify_train_dataset, classify_dev_dataset,trainepoch:int):

        batch_size = 8
        hiddensize = 200
        dataloader = DataLoader(classify_train_dataset, batch_size=batch_size, num_workers=1, collate_fn=collect_fn, shuffle=False)
        total = len(dataloader)
        h0 = torch.FloatTensor(batch_size, 2, hiddensize).cuda(device=self.device_ids[0])
        c0 = torch.FloatTensor(batch_size, 2, hiddensize).cuda(device=self.device_ids[0])
        
        torch.nn.init.kaiming_uniform_(h0,a=0, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(c0,a=0, mode='fan_in', nonlinearity='leaky_relu')
        init_stat = (h0, c0)

        self.bertmodel.train()
        for epoch in range(trainepoch):
            sum= 0
            sum_loss = 0
            for i in dataloader:
                i = self._switch_device(i, self.device_ids[0])
                after = self.bertmodel(encodings=i['encodings'],
                                      context_masks=i['context_masks'],
                                      entity_masks=i['entity_masks'],
                                      init_stat=init_stat)
                after = after.view(-1, after.size(-1))
                loss = self.calssify_loss(after, i['entity_types'].view(-1))
                entity_sample_masks = i['entity_sample_masks'].view(-1).float()
                loss = (loss * entity_sample_masks).sum() / entity_sample_masks.sum()
                sum_loss += loss.item()

                trueY_ =i['entity_types'].view(-1).cpu().detach().numpy().tolist()
                testY_ = after.max(dim=-1)[1].cpu().detach().numpy().tolist()
                trueY =[]
                testY =[]
                for j in range(len(trueY_)):
                    if trueY_[j] != 0:
                        trueY.append(trueY_[j])
                        testY.append(testY_[j])
                F1 = f1_score(trueY, testY, average="micro")
                sum +=F1
                self.calssify_optm.zero_grad()
                loss.backward()
                self.calssify_optm.step()

            # print('F1 : ',sum/len(dataloader) *100,'   loss :  ', sum_loss)


        batch_size = 1
        h0 = torch.FloatTensor(batch_size, 2, hiddensize).cuda(device=self.device_ids[0])
        c0 = torch.FloatTensor(batch_size, 2, hiddensize).cuda(device=self.device_ids[0])
        
        torch.nn.init.kaiming_uniform_(h0,a=0, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(c0,a=0, mode='fan_in', nonlinearity='leaky_relu')
        init_stat = (h0, c0)
        dataloader = DataLoader(classify_dev_dataset, batch_size=batch_size)

        evaluate = Evaluator2(classify_dev_dataset.label_vocab)
        self.bertmodel.eval()
        for i in dataloader:
            i = self._switch_device(i, self.device_ids[0])
            after = self.bertmodel(encodings=i['encodings'],
                                   context_masks=i['context_masks'],
                                   entity_masks=i['entity_masks'],
                                   init_stat=init_stat)
            after = self.softmax(after).view(-1, after.size(-1))

            evaluate.convert_result_to_label(i, after)
        micro = evaluate.score(print_score=False)
        return micro

    def outer_classify_test(self, outer_test_dataset, trainepoch:int):
        batch_size = 1
        hiddensize = 200
        h0 = torch.FloatTensor(batch_size, 2, hiddensize).cuda(device=self.device_ids[0])
        c0 = torch.FloatTensor(batch_size, 2, hiddensize).cuda(device=self.device_ids[0])
        
        torch.nn.init.kaiming_uniform_(h0,a=0, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(c0,a=0, mode='fan_in', nonlinearity='leaky_relu')
        
        
        init_stat = (h0, c0)
        
        self.bertmodel.train()
        dataloader = DataLoader(outer_test_dataset, batch_size=batch_size)
        for epoch in tqdm(range(trainepoch), desc='out_classify training',ncols=80):
            sum= 0
            sum_loss = 0
            for i in dataloader:
                i = self._switch_device(i, self.device_ids[0])
                after = self.bertmodel(encodings=i['encodings'],
                                      context_masks=i['context_masks'],
                                      entity_masks=i['entity_masks'],
                                      init_stat = init_stat)
                after = after.view(-1, after.size(-1))
                loss = self.calssify_loss(after, i['entity_types'].view(-1))
                entity_sample_masks = i['entity_sample_masks'].view(-1).float()
                loss = (loss * entity_sample_masks).sum() / entity_sample_masks.sum()
                sum_loss += loss.item()

                trueY_ =i['entity_types'].view(-1).cpu().detach().numpy().tolist()
                testY_ = after.max(dim=-1)[1].cpu().detach().numpy().tolist()
                trueY =[]
                testY =[]
                for i in range(len(trueY_)):
                    if trueY_[i] != 0:
                        trueY.append(trueY_[i])
                        testY.append(testY_[i])
                F1 = f1_score(trueY, testY, average="micro")
                sum +=F1
                self.calssify_optm.zero_grad()
                loss.backward()
                self.calssify_optm.step()

            print('transfer F1 : ',sum/len(dataloader) *100,'   loss :  ', sum_loss)

    def inner_classify_test(self, inner_test_dataset):
        batch_size = 1
        hiddensize = 200
        h0 = torch.FloatTensor(batch_size, 2, hiddensize).cuda(device=self.device_ids[0])
        c0 = torch.FloatTensor(batch_size, 2, hiddensize).cuda(device=self.device_ids[0])
        
        torch.nn.init.kaiming_uniform_(h0,a=0, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(c0,a=0, mode='fan_in', nonlinearity='leaky_relu')
        init_stat = (h0, c0)
        dataloader = DataLoader(inner_test_dataset, batch_size=batch_size)
        self.bertmodel.eval()
        evaluate = Evaluator2(inner_test_dataset.label_vocab)
        # self.bertmodel = self.bertmodel.eval()
        for i in tqdm(dataloader, total=len(dataloader), desc='in_classify testing',ncols=80):
            i = self._switch_device(i, self.device_ids[0])
            after = self.bertmodel(encodings=i['encodings'],
                                   context_masks=i['context_masks'],
                                   entity_masks=i['entity_masks'],
                                   init_stat = init_stat)
            after = self.softmax(after).view(-1, after.size(-1))

            evaluate.convert_result_to_label(i, after)
        micro = evaluate.score(print_score=True)
        return micro

    def _freeze_model(self, model, freeze = False):
        for weight in model.parameters():
            weight.requires_grad = freeze

    def _init_model(self, model):
        for m in model.parameters():
            init.normal_(m, mean=0, std=1)

    def _switch_device(self, batch, device):
        converted_batch = dict()
        for key in batch.keys():
            converted_batch[key] = batch[key].cuda(device=device)
        return converted_batch

    def _get_optimizer_params(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_params = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        return optimizer_params

    def _load_model(self):
        config = BertConfig.from_pretrained(self.args.tokenizer_path)

        model = Bertmodel.from_pretrained(self.args.tokenizer_path,
                                            config=config,
                                            tokenizer_shape = self.args.tokenizer_shape,
                                            out_shape = self.args.ways+1,
                                            reduct_shape = self.x
                                            # freeze_transformer = False
                                      )

        return model






#
#