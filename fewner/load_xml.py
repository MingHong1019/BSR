
import json


def get_onto_sentence(file_path):
    f = open(file_path, 'r', encoding='utf-8')
    file = f.read()
    f.close()
    lines = file.split('\n')
    sentences = []
    types = []
    sen = []
    label = []
    for line in lines:
        if line == '':
            sentences.append(sen)
            types.append(label)
            sen = []
            label = []
        else:
            line = line.split()
            sen.append(line[0])
            label.append(line[1])
    return sentences, types

def get_all_onto_sentence(onto_path:str):
    sentences, types = [], []
    result = []
    for i in [r'/test', r'/train', r'/valid']:
        a, b = get_onto_sentence(onto_path+i+'.txt')
        sentences+=a
        types+=b

    result = []
    total_type=[]
    for i in range(len(sentences)):
        token = sentences[i]
        label_start = [j[0] for j in types[i]] + ['O']
        B_idx = [j for j in range(len(label_start)) if label_start[j] == 'B']
        # label_ = [label_vocab[label[i][2:]] for i in B_idx]

        type = []
        for m in B_idx:
            start = m
            end = m + 1
            for j in range(end, len(token)):
                if label_start[j] != 'I':
                    break
                end = j
            entity_type = types[i][start][2:]
            if entity_type not in total_type:
                total_type.append(entity_type)
            type.append( dict(start=start, end=end, type=entity_type))
        result.append(dict(sentence=token, type=type))
    return result, total_type


def read_json(json_path):
    f = open(json_path + r'/dev.json')
    file = json.load(f)
    f.close()
    f = open(json_path + r'/test.json')
    file += json.load(f)
    f.close()
    f = open(json_path + r'/train.json')
    file += json.load(f)
    f.close()
    entity_type = []
    for i in file:
        for j in i['type']:
            if j['type'] not in entity_type:
                entity_type.append(j['type'])

    return file, entity_type



if __name__ == '__main__':
    test = r'data/ontonotes'

    entity_num =0
    entity_type = []
    a,b = get_all_onto_sentence(test)

    print()


    # print(res.__len__())



