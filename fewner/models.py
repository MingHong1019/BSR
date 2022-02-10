import torch
import numpy as np
from torch import nn as nn
from transformers import BertConfig
from transformers import BertModel
from transformers import BertPreTrainedModel
from torch.nn.utils.rnn import pack_padded_sequence,pack_sequence,pad_packed_sequence

from transformers import BertConfig
from transformers import BertModel
from transformers import BertPreTrainedModel



class Bertmodel(BertPreTrainedModel):
    def __init__(self, config: BertConfig, tokenizer_shape, reduct_shape, out_shape):
        super(Bertmodel, self).__init__(config)

        self.bert = BertModel(config)
        self.hidden_size = hidden_size = 200
        self.reduct_shape = reduct_shape
        self.bilstm = nn.LSTM(input_size=tokenizer_shape, hidden_size=hidden_size, 
                              batch_first=True, bidirectional = True)

        self.classify = nn.Linear(reduct_shape, out_shape)
        self.U = nn.Parameter(torch.FloatTensor(self.hidden_size, reduct_shape, self.hidden_size))
        torch.nn.init.kaiming_uniform_(self.U, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(2 * self.hidden_size, reduct_shape)
        self.norm = nn.LayerNorm(reduct_shape, elementwise_affine=False)
        # self.init_weights()

    def get_span_represent(self, encodings: torch.tensor,context_masks: torch.tensor, 
                           entity_masks: torch.tensor, init_stat):
        h = self.bert(input_ids=encodings, attention_mask=context_masks)['last_hidden_state']
        total_len = h.size(1)
        h = context_masks.float().unsqueeze(-1)* h
        
        seq_len = context_masks.float().sum(dim=-1).cpu()
        h = pack_padded_sequence(h, seq_len, batch_first=True, enforce_sorted =False)
        h0 = init_stat[0].transpose(0,1)
        c0 = init_stat[1].transpose(0,1)
        h = self.bilstm(h,(h0, c0))[0]
       
        h, result_len = pad_packed_sequence(h, batch_first=True, total_length=total_len)
        batch_size = h.size(0)
        hs, he = h[:, :, :self.hidden_size].reshape(-1, self.hidden_size), h[:, :, self.hidden_size:].reshape(-1,self.hidden_size)
        result1 = self.U @ he.transpose(-1, -2)
        result1 = hs @ result1.view(-1, self.hidden_size, self.reduct_shape)
        result2 = self.linear(torch.cat((hs, he), dim=-1))
        result1 = result1 + result2
        h = result1.mean(dim=0)
        h = h.view(batch_size, -1, h.size(1))

        m = (entity_masks.unsqueeze(-1) == 0).float() * (-1e30)
        entity_spans_pool = m + h.unsqueeze(1).repeat(1, entity_masks.shape[1], 1, 1)
        entity_spans_pool = entity_spans_pool.max(dim=2)[0]
        return entity_spans_pool

  
    def forward(self, encodings: torch.tensor, context_masks: torch.tensor,
                  entity_masks: torch.tensor, init_stat):
        self.bilstm.flatten_parameters()
        result = self.get_span_represent(encodings, context_masks, entity_masks, init_stat)
        result = self.norm(result)
        result = self.dropout(result)
        result = self.classify(result)
        return result



