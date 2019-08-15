import json
from keras_bert import Tokenizer
import codecs
import numpy as np
from trie import *
from data_util import new_alias,del_bookname
def entity_clear(text):
    pun = {'，': ',',
           '·': '•',
           '：': ':',
           '！': '!',
           }
    for p in pun:
        if p in text:
            text=text.replace(p,pun[p])
    return text.lower()
def get_entity_id():
    '''
    与data_util 里面得到大致entity_id相同，区别是 为了更全的匹配mention，将实体名字全部变成小写
    :return:
    '''
    new_entity_alias=new_alias()
    entity_id={}
    with open('original_data/kb_data', 'r') as f:
        for line in f:
            temDict = json.loads(line)
            subject=temDict['subject']
            subject_id=temDict['subject_id']
            alias = set()
            for a in temDict['alias']:
                alias.add(a.lower())
            alias.add(entity_clear(subject))
            if subject in new_entity_alias:
                alias=alias|new_entity_alias[subject]
            alias.add(subject.lower())
            entity_set=set()
            for en in alias:
                entity_set.add(en.lower())
            for n in entity_set:
                n=del_bookname(n)
                if n in entity_id:
                    entity_id[n].append(subject_id)
                else:
                    entity_id[n]=[]
                    entity_id[n].append(subject_id)
    return entity_id

def entity_embedding():
    '''
    得到实体名字的嵌入
    :return:
    '''
    entity_id=get_entity_id()
    print(len(entity_id))
    id_embedding = pd.read_pickle('data/id_embedding.pkl')
    entity_index = {}
    embedding_matrix = np.random.normal(size= (len(entity_id)+1, 768))
    for i, en in enumerate(entity_id):
        entity_index[en] = i+1
        vec=[]
        for id in entity_id[en]:
            vec.append(id_embedding[id])
        vec=np.mean(vec,axis=0)
        embedding_matrix[i+1]=vec
    pd.to_pickle(entity_index,'data/ER_entity_index.pkl')
    np.save('data/ER_entity_embedding.npy',embedding_matrix)
    print(embedding_matrix.shape)#(313864, 768)


def seq_padding(seq,max_len,value=0):
    x=[value]*max_len
    x[:len(seq)]=seq[:max_len]
    return x

def get_token_dict():
    bert_path = 'bert_model/'
    dict_path = bert_path+'vocab.txt'
    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict
def get_ids_seg(text,tokenizer,max_len=52):
    indices, segments = [], []
    for ch in text[:max_len-2]:
        indice, segment = tokenizer.encode(first=ch)
        if len(indice) != 3:
            indices += [100]
            segments += [0]
        else:
            indices += indice[1:-1]
            segments += segment[1:-1]
    segments = [0] + segments + [0]
    indices = [101] + indices + [102]
    return indices,segments

def get_match_entity(match_en,split_entity):
    '''
    处理可分割实体
    :param match_en:
    :param split_entity:
    :return:
    '''
    match_entity = []
    for en in match_en:
        if en[0] in split_entity:
            for e in split_entity[en[0] ]:
                match_entity.append((e,en[0].find(e)+en[1]))
        else:
            match_entity.append(en)
    return match_entity

def get_input_bert(input_file,out_file,max_len=52):
    '''
    构建模型的训练集输入
    :param input_file:
    :param out_file:
    :param max_len:
    :return:
    '''
    token_dict = get_token_dict()
    tokenizer = Tokenizer(token_dict)
    split_entity = pd.read_pickle('data/split_entity.pkl')

    trie_obj = get_Trie()
    entity_index = pd.read_pickle('data/ER_entity_index.pkl')
    print(len(entity_index))
    inputs = {'ids': [], 'seg': [], 'begin': [], 'end': [],'labels': [], 'entity_id': [], 'match_en': []}
    match_len=0
    with open(input_file, 'r') as f:
        for line in f:
            temDict = json.loads(line)
            text = temDict['text']
            indices, segments = get_ids_seg(text, tokenizer)
            mention_data = temDict['mention_data']
            inputs['ids'].append(seq_padding(indices, max_len))
            inputs['seg'].append(seq_padding(segments, max_len))
            match_en = trie_obj.search_entity(text)
            if len(match_en)>match_len:
                match_len=len(match_en)
            match_en = get_match_entity(match_en,split_entity)
            label2 = [0] * len(match_en)

            en_list = []
            for en in match_en:
                if en[0] in entity_index:
                    en_list.append(entity_index[en[0]])
                else:
                    en_list.append(0)
                    print(en[0])
            begin = []
            end = []
            men_set = set()

            for men in mention_data:
                if men['kb_id'] != 'NIL':
                    men_set.add((men['mention'], int(men['offset'])))
            text = temDict['text']
            for i, en in enumerate(match_en):
                begin.append(en[1] + 1)
                end.append(en[1] + len(en[0]))
                if (text[en[1]:en[1] + len(en[0])], en[1]) in men_set:
                    label2[i] = 1
            inputs['labels'].append(seq_padding(label2, 13,value=0))
            inputs['begin'].append(seq_padding(begin, 13, value=-1))
            inputs['end'].append(seq_padding(end, 13, value=-1))
            inputs['entity_id'].append(seq_padding(en_list, 13))
            inputs['match_en'].append(match_en)

    for key in inputs:
        inputs[key] = np.array(inputs[key])
        print(key, inputs[key].shape)
        print(inputs[key][0])
    pd.to_pickle(inputs, out_file)

    print(match_len)
def get_input_bert_test(input_file,out_file,max_len=52):
    '''
    构建模型测试集输入
    :param input_file:
    :param out_file:
    :param max_len:
    :return:
    '''
    token_dict = get_token_dict()
    tokenizer = Tokenizer(token_dict)
    trie_obj = get_Trie()
    split_entity = pd.read_pickle('data/split_entity.pkl')
    entity_index = pd.read_pickle('data/ER_entity_index.pkl')
    inputs = {'ids': [], 'seg': [], 'begin': [], 'end': [], 'labels': [0], 'entity_id': [],
              'match_en': []}

    with open(input_file, 'r') as f:
        for line in f:
            temDict = json.loads(line)
            text = temDict['text']
            indices, segments = get_ids_seg(text, tokenizer)
            inputs['ids'].append(seq_padding(indices, max_len))
            inputs['seg'].append(seq_padding(segments, max_len))
            match_en = trie_obj.search_entity(text)
            match_en = get_match_entity(match_en, split_entity)
            en_list = [entity_index[en[0]] for en in match_en]
            begin = []
            end = []
            for i, en in enumerate(match_en):
                begin.append(en[1] + 1)
                end.append(en[1] + len(en[0]))
            inputs['begin'].append(seq_padding(begin, 13, value=-1))
            inputs['end'].append(seq_padding(end, 13, value=-1))
            inputs['entity_id'].append(seq_padding(en_list, 13))
            inputs['match_en'].append(match_en)
    for key in inputs:
        inputs[key] = np.array(inputs[key])
        print(key, inputs[key].shape)
        print(inputs[key][0])
    pd.to_pickle(inputs, out_file)

if __name__ == '__main__':
    entity_embedding()
    get_input_bert('original_data/train.json','data/ER_input_train_match.pkl')
    get_input_bert_test('original_data/eval722.json', 'data/input_eval_match_bert.pkl')


    pass
