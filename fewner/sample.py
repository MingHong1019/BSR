
import torch
from transformers import BertTokenizer
import random

def create_sample(doc:dict, max_span_size=7, neg_entity_count=200, neg_sample=True):
    encoding = doc['sentence']
    token_count = len(doc['token_idx'])
    context_size = len(encoding)

    # positive entities
    pos_entity_spans, pos_entity_types, pos_entity_masks = [], [], []
    for i in range(len(doc['span'])):
        pos_entity_spans.append(doc['span'][i])
        pos_entity_types.append(doc['label'][i])
        pos_entity_masks.append(create_entity_mask(doc['span'][i][0],doc['span'][i][1], context_size))

    # negative entities
    neg_entity_spans, neg_entity_masks = [], []
    for size in range(1, max_span_size + 1):
        for i in range(0, (token_count - size)):
            # if doc['token_idx'][i]==75:
            #     print()
            span = [doc['token_idx'][i], doc['token_idx'][i + size]]
            if span not in pos_entity_spans:
                neg_entity_spans.append(span)
    if neg_sample:
        neg_entity_spans = random.sample(neg_entity_spans, min(len(neg_entity_spans),neg_entity_count))
    neg_entity_masks = get_span_mask(neg_entity_spans, context_size)
    neg_entity_types = [0 for _ in range(len(neg_entity_spans))]

    # merge
    entity_types = pos_entity_types + neg_entity_types
    entity_masks = pos_entity_masks + neg_entity_masks

    encodings = torch.tensor(encoding, dtype=torch.long)
    context_masks = torch.ones(context_size, dtype=torch.bool)
    entity_types = torch.tensor(entity_types, dtype=torch.long)
    entity_masks = torch.stack(entity_masks)
    pos_entity_masks = torch.stack(pos_entity_masks)
    pos_entity_types = torch.tensor(pos_entity_types, dtype=torch.long)
    token_idx = torch.tensor(doc['token_idx'], dtype=torch.long)
    entity_sample_masks = torch.ones([entity_masks.shape[0]], dtype=torch.bool)

    return dict(encodings=encodings, context_masks=context_masks,
                entity_masks=entity_masks,entity_types=entity_types,
                pos_masks=pos_entity_masks, pos_type=pos_entity_types,
                token_idx = token_idx, entity_sample_masks=entity_sample_masks
                )


#
# def create_sample(sentence: list, label: list, label_vocab:dict,
#                   tokenizer: BertTokenizer, neg_sample_num=100, neg_sample=False):
#     word_index = []
#     doc_encoding = [tokenizer.convert_tokens_to_ids('[CLS]')]
#     index = 1
#     for i in sentence:
#         word_encode = tokenizer.encode(i, add_special_tokens=False)
#         word_index.append(index)
#         index += len(word_encode)
#         doc_encoding += word_encode
#     doc_encoding += [tokenizer.convert_tokens_to_ids('[SEP]')]
#     word_index.append(index)
#
#     entity_span, label_span = get_entity_span(label, label_vocab, word_index)
#     span_mask = get_span_mask(entity_span, len(doc_encoding))
#     sentence_mask = torch.ones(len(doc_encoding), dtype=torch.bool)
#     doc_encoding = torch.tensor(doc_encoding, dtype=torch.long)
#     word_index = torch.tensor(word_index, dtype=torch.int)
#     if neg_sample:
#         max_span_size = 7
#         max_san_num = neg_sample_num
#         # max_sample_size = max(max_span_size, len(word_index)-1)
#         neg_entity_spans, neg_entity_mask = [], []
#         for size in range(1, max_span_size + 1):
#             for i in range(0, (len(word_index) - size)):
#                 span = [word_index[i], word_index[i + size]]
#                 if span in entity_span:
#                     continue
#                 else:
#                     neg_entity_spans.append(span)
#
#         if max_san_num != 0:
#             max_san_num = min(max_san_num, len(neg_entity_spans))
#             neg_entity_spans = random.sample(neg_entity_spans, max_san_num)
#
#         neg_entity_mask = get_span_mask(neg_entity_spans, len(doc_encoding))
#         neg_label = [0 for i in  neg_entity_mask]
#         span_mask = torch.stack(span_mask + neg_entity_mask)
#         label_span = torch.tensor(label_span + neg_label, dtype=torch.long)
#
#
#     else:
#         if label_span != []:
#             span_mask = torch.stack(span_mask)
#             label_span = torch.tensor(label_span, dtype=torch.long)
#         else:
#             span_mask = torch.zeros(len(doc_encoding), dtype=torch.bool).view(1,-1)
#             label_span = torch.tensor((0), dtype=torch.long).view(-1)
#
#     return dict(encodings=doc_encoding, entity_masks=span_mask,
#                 entity_types=label_span, context_masks=sentence_mask,
#                 token_idx=word_index
#                 )
#
#
# def get_entity_span(label:list, label_vocab:dict, word_index:list):
#     span = []
#     label_span = []
#     label_start = [i[:1] for i in label] + ['O']
#     label_end = [i[2:] for i in label] + ['']
#     B_index = [i for i, x in enumerate(label_start) if x == 'B']
#     for i in B_index:
#         for j in range(i+1, len(label_start)):
#             if label_start[j] == 'B' or label_start[j] == 'O':
#                 span.append([word_index[i], word_index[j]])
#                 m = label_end[i]
#                 label_span.append(label_vocab[m])
#                 break
#     return span, label_span
#

def create_entity_mask(start, end, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end] = 1
    return mask

def get_span_mask(entity_span:list, context_size:int):
    span_mask = []
    for i in entity_span:
        mask = torch.zeros(context_size, dtype=torch.bool)
        mask[i[0]:i[1]] = 1
        span_mask.append(mask[:])
    return span_mask


def extend_tensor(tensor, extended_shape, fill=0):
    tensor_shape = tensor.shape

    extended_tensor = torch.zeros(extended_shape, dtype=tensor.dtype).to(tensor.device)
    extended_tensor = extended_tensor.fill_(fill)

    if len(tensor_shape) == 1:
        extended_tensor[:tensor_shape[0]] = tensor
    elif len(tensor_shape) == 2:
        extended_tensor[:tensor_shape[0], :tensor_shape[1]] = tensor
    elif len(tensor_shape) == 3:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2]] = tensor
    elif len(tensor_shape) == 4:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2], :tensor_shape[3]] = tensor

    return extended_tensor


def padded_stack(tensors, padding=0):
    dim_count = len(tensors[0].shape)

    max_shape = [max([t.shape[d] for t in tensors]) for d in range(dim_count)]
    padded_tensors = []

    for t in tensors:
        e = extend_tensor(t, max_shape, fill=padding)
        padded_tensors.append(e)

    stacked = torch.stack(padded_tensors)
    return stacked

def collect_fn(batch:dict):
    padded_batch = dict()
    keys = batch[0].keys()

    for key in keys:
        samples = [s[key] for s in batch]

        if not batch[0][key].shape:
            padded_batch[key] = torch.stack(samples)
        else:
            padded_batch[key] = padded_stack([s[key] for s in batch])

    return padded_batch


#
# sentence = ['I','embedding','a','word']
# label = ['B-PER','I-PER','O','B-PER']
# label_vocab = {'PER':1}
# tokenizer_path = 'bert-base-cased'
# tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=False)
# a = create_sample(sentence, label, label_vocab, tokenizer, neg_sample=True)
# print()