import json
import codecs
from keras_bert import Tokenizer
import pandas as pd
import numpy as np
from data_util import *
from trie import *
import random
from keras_bert import Tokenizer

def get_link_entity(mention,id,entity_id,id_entity):
    '''
    1. 先通过mention 得到要消歧的所有实体id，
    2. 弱通过mention 找不到，则通过对应id，根据id找到对应实体，然后在找到要消歧的所有实体id，
    :param mention: 训练集中mention
    :param id: mention对应id
    :param entity_id: 实体名字对应所有实体的id
    :param id_entity: id对应的实体名字
    :return:去掉正确实体id的，id列表
    '''
    link_entitys=[]
    if mention in entity_id:
        link_entitys+=list(entity_id[mention])
        link_entitys+=list(entity_id[id_entity[id]])
    else:
        link_entitys+=list(entity_id[id_entity[id]])
    link_entitys=set(link_entitys)
    link_entitys.remove(id)
    link_entitys=list(link_entitys)
    random.shuffle(link_entitys)
    return link_entitys[:3]

def get_link_entity_test(mention,entity_id):
    if mention in entity_id:
        return list(entity_id[mention])
    return []

def get_token_dict():
    bert_path = 'bert_model/'
    dict_path = bert_path+'vocab.txt'
    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict
def get_input():
    '''
    消歧输入的构建
    :return:
    '''
    id_type=pd.read_pickle('data/id_type.pkl')
    type_index=pd.read_pickle('data/type_index.pkl')
    entity_id = pd.read_pickle('data/entity_id.pkl')
    id_entity=pd.read_pickle('data/id_entity.pkl')
    id_text=pd.read_pickle('data/id_text.pkl')
    inputs = {'ids': [], 'seg': [],'begin':[],'end':[],'en_type':[],'labels':[]}
    token_dict = get_token_dict()
    tokenizer = Tokenizer(token_dict)
    file_len=0
    trie_obj=get_Trie()
    with open('original_data/train.json') as f:
        for line in f:
            if file_len%100==0:
                print(file_len)
            file_len+=1
            temDict = json.loads(line)
            text=temDict['text']
            match_en = trie_obj.search_entity(text)
            mention_data=temDict['mention_data']
            for men in mention_data:
                mention=men['mention']
                kb_id=men['kb_id']
                offset = men['offset']
                begin=int(offset)+1
                end = begin+len(mention)
                if kb_id != 'NIL':
                    link_id=[kb_id]
                    link_id+=get_link_entity(mention,kb_id,entity_id,id_entity)

                    for id in link_id:
                        kb_text = id_text[id]
                        kb_type=type_index[id_type[id][0]]
                        indice, segment = tokenizer.encode(first=text,second=kb_text,max_len=256)
                        inputs['ids'].append(indice)
                        inputs['seg'].append(segment)
                        inputs['begin'].append([begin])
                        inputs['end'].append([end])
                        if id ==kb_id:
                            inputs['labels'].append(1)
                        else:
                            inputs['labels'].append(0)
                        inputs['en_type'].append([kb_type])
            mention_set=set()
            for men in mention_data:
                mention_set.add((men['mention'],int(men['offset'])))
            for en in match_en:
                if not en in mention_set:

                    link_id = get_link_entity_test(en[0],entity_id)
                    for id in link_id[:1]:
                        kb_text = id_text[id]
                        kb_type = type_index[id_type[id][0]]
                        indice, segment = tokenizer.encode(first=text, second=kb_text, max_len=256)
                        inputs['ids'].append(indice)
                        inputs['seg'].append(segment)
                        inputs['begin'].append([begin])
                        inputs['end'].append([end])
                        inputs['labels'].append(0)
                        inputs['en_type'].append([kb_type])
                    break

    for k in inputs:
        inputs[k]=np.array(inputs[k])
        print(k,inputs[k].shape)
        print(inputs[k][1])
    pd.to_pickle(inputs,'data/train_input_bert_final.pkl')


def get_test_input(input_file, out_file):
    id_type = pd.read_pickle('data/id_type.pkl')
    type_index = pd.read_pickle('data/type_index.pkl')
    entity_id = pd.read_pickle('data/entity_id.pkl')

    id_text = pd.read_pickle('data/id_text.pkl')

    token_dict = get_token_dict()
    tokenizer = Tokenizer(token_dict)
    out_file = open(out_file, 'w')
    file_index = 0
    with open(input_file) as f:
        for line in f:
            if file_index%100==0:
                print(file_index)
            file_index+=1

            temDict = json.loads(line)
            text = temDict['text']
            mention_data = temDict['mention_data']
            for men in mention_data:
                mention = men['mention']

                offset = int(men['offset'])
                begin = int(offset)+1
                end = begin + len(mention)

                link_id = get_link_entity_test(mention, entity_id)
                men['link_id'] = link_id
                link_data = {'ids': [], 'seg': [],'begin':[],'end':[],'en_type':[]}
                for id in link_id:

                    kb_text = id_text[id]
                    kb_type = type_index[id_type[id][0]]
                    indice, segment = tokenizer.encode(first=text, second=kb_text, max_len=256)
                    link_data['ids'].append(indice)
                    link_data['seg'].append(segment)
                    link_data['begin'].append([begin])
                    link_data['end'].append([end])
                    link_data['en_type'].append([kb_type])
                men['link_data'] = link_data

            out_file.write(json.dumps(temDict, ensure_ascii=False))
            out_file.write('\n')

if __name__ == '__main__':
    get_input()
    # get_test_input('data/dev.json', 'data/dev_link_binary_true_bert.json')

    # get_test_input('result/dev_ner_result.json','data/dev_link_binary_bert.json')
    #
    # get_test_input('result/test_ner_result.json', 'data/test_link_binary_bert.json')

    get_test_input('result/eval_ner_result.json', 'data/eval_link_binary_bert.json')

    pass



